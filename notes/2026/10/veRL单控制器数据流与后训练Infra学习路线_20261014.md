# veRL 单控制器数据流与后训练 Infra 学习路线

> 日期：2026-10-14
> 来源：[当日 Inbox](../../../inbox/2026-10-14.md)；只保留开源框架 veRL 的公开设计、算法需求到 Infra 设计的映射和学习方法，不记录个人职业安排与内部工作上下文。
> 状态：已整理

## 整理记录

- 2026-10-14 18:52：Claude Code 自动整理（当天主整理）；来源 Claude Code 会话 `684a7fe5`（14:42–15:02）、`10f7bbbd`（15:06）。

> 证据边界：veRL 迭代很快。讲解时拉取的最新 main 已删除经典的 FSDP worker 与 sharding manager 文件，经典 Trainer 也标为 deprecated；下文以结构最完整的 v0.5.0 为准，具体函数名请对照该版本源码。

## 0. 今天最终要解决的问题

1. veRL 的 Single-Controller 与 SPMD 混合编程模型怎么落地？`DataProto` 和 `fit()` 主循环在做什么？
2. 用户的追问：Infra 一定要结合算法——能不能先设计一个从零开始、由算法目的过渡到 Infra 实现的读源码计划？
3. 一份“后训练 Infra 五大能力”的说法，哪些对、哪些需要修正？

## 1. 先给结论

- **Driver 进程手里有整个 batch（在 CPU 上）**。代码里每个 `xxx_wg.method(batch)` 都是一次 RPC：按 world size 切开 → 分发给 N 个 GPU Worker → 各 Worker SPMD 同时执行 → 收回 Driver 拼成一个。Driver 自己只做 Advantage 这类轻计算。所以 `fit()` 读起来像单机代码。
- **`DataProto` 是节点间传输的统一格式**：按第 0 维切的 Tensor、共用同一切分边界的非 Tensor、不切分原样广播的 meta。
- **读源码要沿着“算法需求 → 工程问题 → 框架解法”**：每个 Infra 设计都对应一个算法需求，脱离算法读抽象层很快就会忘。
- **能力清单的三处校准**：大模型权重同步实际是秒级，难点是正确性与峰值显存，不是压到毫秒；推训精度偏差现在多默认存在、用重要性采样一类方法在算法上修正，而不是强行逐位对齐；“怎么测量一步 RL 的时间与偏差”本身是核心能力。

## 2. 前置概念

### 2.1 Single-Controller 与 SPMD

| | 含义 | 在 veRL 里 |
|---|---|---|
| Single-Controller | 一个进程持有全局控制流，按顺序调用各角色 | Driver 上的 `fit()` |
| SPMD（多控制器） | 每个 rank 跑同一份程序、处理自己那份数据 | 每个 GPU Worker 内部的训练/推理引擎 |

混合模型：控制流集中、计算分布。改算法逻辑（KL 放哪、baseline 怎么算）只需改 Driver 上的单机式代码。

### 2.2 和 Ray 的关系

`wg.method()` 背后是 Ray actor 的远程调用；Worker 是持有本机 GPU 的进程（见 [Ray 进程模型](Ray进程模型对象引用生命周期与重算队列_20261013.md)）。

## 3. 完整原理

### 3.1 `DataProto` 的三块

| 字段 | 切分方式 | 用途 | 容易出问题的地方 |
|---|---|---|---|
| `batch`（TensorDict） | 第 0 维连续 chunk | 训练与推理的主要数据，如 `input_ids [B, L]`、`old_log_probs [B, R]` | Driver 上是 padding 后的二维张量，padding 也一起被序列化发出去 |
| `non_tensor_batch` | 与 `batch` 共用切分边界 | `uid`、原始 prompt、图片、工具参数 | 必须是 `dtype=object` 的 ndarray，长度不对会在一致性检查里报错 |
| `meta_info` | 不切，每份给完整一份 | 温度、micro batch 大小等控制参数，以及 Worker 返回的 metrics | 拼回时同名非 metrics key 若各 Worker 值不同会触发断言 |

几个关键方法：

- `chunk`（Scatter 的核心）：按第 0 维切成 world size 份；**不能整除时直接报错**，所以分发层会先做 auto padding；非 Tensor 部分用 Tensor 部分的切分边界来切，保证每一行对得上；
- `concat`（Gather 的核心）：在 Driver 的 CPU 内存里拼回完整 batch；metrics 合并成 dict-of-list；
- `union`：主循环里不断“往 batch 上加列”；要求 batch size 完全相同，**同名 key 会做全量相等比较**，大张量反复 union 有开销；
- `repeat`：GRPO 的 n 次采样 `[p0, p0, …, p1, p1, …]`，Driver 内存 × n；
- 序列化（发给 Worker 或存进对象存储时）：把所有 Tensor 合到一块连续 buffer 再保存——**一次完整的内存拷贝 + 序列化，batch 有几个 GB 时 Driver 会卡在这里**。

`DataProtoFuture`：存一组 ObjectRef 加一个分发函数，让上一个 WorkerGroup 的输出**不经过 Driver 的 get** 直接作为下一个的输入。限制是在 Driver 上不能对它做任何操作；而经典同步 `fit()` 每一步都要在 Driver 上 union、算 Advantage，所以它几乎用不上——这是为异步流水线准备的设计。

### 3.2 `fit()` 主循环骨架

```python
for batch_dict in train_dataloader:
    batch = DataProto.from_single_dict(batch_dict)       # [B] prompt，左 padding，Driver CPU
    gen_batch = batch.pop(["input_ids", "attention_mask", "position_ids"], ...)
    gen_batch = gen_batch.repeat(rollout.n, interleave=True)   # [B*n]

    gen_out = actor_rollout_wg.generate_sequences(gen_batch)
    # Scatter → Worker 内切到推理态、唤醒推理引擎 → 生成 → 回到训练态 → Gather
    # 整步最耗时；长尾样本决定这一步何时结束

    batch.non_tensor_batch["uid"] = 每个 prompt 一个 uuid   # GRPO 靠它分组
    batch = batch.repeat(rollout.n, interleave=True)        # 必须与 gen_out 同样 interleave，否则行对不上
    batch = batch.union(gen_out)
    # 之后：参考策略 logprob → 旧策略 logprob 重算 →（PPO）Critic values
    # → Driver 上算 Reward / KL / Advantage → update_critic → update_actor
```

一步里每个 `wg.` 调用都是一整轮分发与收集；batch 的列一路增加。

### 3.3 算法需求如何决定 Infra 设计

| 算法需求 | 带来的工程问题 | 框架解法 |
|---|---|---|
| 一步 PPO 要 4 个模型、5 类计算（生成、3 次无梯度前向、2 次反向） | 同时放显存放不下，分开放又有 GPU 闲置 | 角色共置（colocate）+ 参数 offload |
| 算法逻辑改得很频繁 | 控制流必须能像单机代码一样改 | Single-Controller 驱动 SPMD Worker |
| On-policy：每步用当前策略采样 | 每步都要把训练权重同步给推理引擎，两边显存形态完全不同 | Sharding Manager + 推理引擎 sleep/wake |
| `ratio = π_θ/π_old`、Loss 用 token-mean | 切 micro-batch 与 DP 后分母和数值一致性出问题 | 全局 token 数归一化、重算 old logprob |
| Response 长度方差很大 | padding 浪费算力，rank 间负载不均（木桶效应） | 序列长度均衡、去 padding、动态 batch、序列并行 |
| 多轮对话与工具调用 | 生成不再是一次性 batch generate，同步模式下 GPU 空转 | Agent loop、服务化 rollout、异步训练 |

第四行与 [全局 Token 归一化](../09/MTP全局Token归一化与多深度梯度_20260911.md) 是同一个问题：**切分后仍按局部 token 数求平均，梯度会被悄悄算错**——这是算法正确性第一次依赖 Infra 实现。

### 3.4 读源码计划的结构

用户要求“由上至下、由浅入深，先讲算法目的，再讲 Infra 在做什么”。计划分三层，每个阶段都按四步：**算法要算什么 → 由此产生什么工程约束 → 框架在哪段代码解决 → 动手验证**。

| 层 | 阶段 | 读完要能回答 / 产出 |
|---|---|---|
| 算法要算什么 | 0 入口与配置地图 | 各配置项作用在算法层、Worker 层还是推理引擎层 |
| | 1 算法契约 | 一张张量清单：名字、shape、dtype、谁产生、谁使用、要不要梯度；用小张量验证 micro-batch 直接平均 ≠ 全局平均 |
| | 2 一步训练的数据流 | `DataProto` 与 `fit()`（即 §3.1～3.2） |
| Infra 怎么让算法跑起来 | 3 进程与 GPU 布局 | 资源池、多个 Worker 类合并进同一个 Ray actor 共享 GPU 与 CUDA context |
| | 4 单控制器驱动 SPMD | 注册装饰器与分发模式如何把一次调用变成 dispatch → 执行 → collect；本地 CPU Ray 写玩具 Worker 验证 |
| | 5 Worker 内多角色 | Actor 与 Ref 复用底座、三层 batch 循环、offload 时机；纸面算显存账 |
| | 6 推训切换（重点） | 进入时唤醒推理引擎、收集分片权重、加载、重建 KV；退出时推理引擎 sleep、显存还给训练；画单卡显存时间线、算每步权重同步数据量 |
| 效率、正确性与前沿 | 7 吞吐与显存 | 序列长度均衡、去 padding、动态 batch 下 Loss 缩放是否仍对 |
| | 8 起往后 | 推训数值对齐、main 上的重构与异步/多轮 Agent |

原则：版本以 v0.5.0 为主线，main 上的重构放在最后当作“为什么这样演进”来读；动手题尽量能在 CPU 上做，只有 GPU 才能验证的改成纸面推演。

### 3.5 对“后训练 Infra 五大能力”的校准

用户贴来一份外部回答：双栈功底（训练栈 + 推理栈 + 权重重分片）、异构调度与数据流、显存压榨与算子、算法敏感度与数值对齐、RLVR/Agent 环境基建。助手的判断：作为**岗位能力描述**基本对，但不能直接当学习计划——五个方向并行学，容易每个都只懂到概念。

| 说法 | 校准 |
|---|---|
| 推训一体与数值对齐是核心壁垒 | 对：这是后训练与预训练 Infra 真正不同的地方，也最难靠读文章学会 |
| Logprob 显存洪峰与 sequence packing 必须掌握 | 对：每天都会碰到（参见 [MTP 显存生命周期](../08/MTP显存生命周期与Allocator性能归因_20260828.md)） |
| 权重同步可以做到毫秒级 | 夸张：大模型实际一般秒级，难点是 layout/dtype 对得上和峰值显存不炸 |
| 推训精度偏差要逐层对齐到完全一样 | 重点已变：默认两边一定有偏差，转而在算法上用重要性采样修正（TIS/MIS 一类）；MoE 还有路由不一致（路由回放） |
| 框架只提 OpenRLHF 与 veRL | 还有基于 Megatron + SGLang 的精简框架与全异步框架，前者代码精简、适合读 |
| 沙箱与 Agent 环境 | 更接近通用后端工程，可排后 |
| （缺失） | **“怎么衡量”**：一步 RL 的时间分解、吞吐、Logprob 偏差怎么量，本身是核心能力 |

学习计划的三条原则：**每阶段都要有产出**（文档、代码或一组数字）；**先测量再读代码**（先看时间花在哪，再带着问题读）；**尽量挨着手头工作选题**。性价比最高的一步是**自己从零写一个约千行的 nano-GRPO**（训练侧 + 推理引擎 rollout，自己实现权重同步、Loss Mask、Advantage）——亲手撞上 off-by-one 的 mask、两侧 Logprob 对不上、权重同步时显存翻倍、长度不齐导致 GPU 空等，再回去读框架就知道每层抽象在解决什么。

## 4. 与已有知识的联系

- 与 [训练部署分层](../07/训练部署Baseline与RLVR闭环_20260727.md#2-先把系统分层每层只回答一种问题) 相关：那里的“训练编排层”在 veRL 里就是 Driver 上的 `fit()`。
- 与 [PPO 九段流程](PPO的Critic与Clip在单次更新下的退化_20261013.md#23-ppo-一步的九段流程) 相关：`fit()` 是那九段在代码里的样子。
- 与 [推理引擎的 KV 分页与 CUDA Graph](推理引擎的KV分页连续批处理与CUDAGraph_20261014.md#39-rl-与这三层的耦合) 相关：推训切换时的 sleep/wake、权重原地更新、清前缀缓存。
- 与 [Ray 进程模型](Ray进程模型对象引用生命周期与重算队列_20261013.md) 相关：`DataProtoFuture` 就是“只传提货单、不经 Driver 取货”。

## 5. 工程中的具体表现

| 现象 | 原因 | 如何验证 |
|---|---|---|
| Driver 在分发前卡几秒 | 大 batch 整块拷贝 + 序列化 | 对 Driver 做 CPU profile；看 batch 体积 |
| 分发时报不能整除 | batch 大小不是 world size 的倍数 | 检查 auto padding 是否开启 |
| union 报行数或值不一致 | repeat 的 interleave 方式不一致，或同名 key 值不同 | 打印两侧 batch size 与 uid 顺序 |
| 切 micro-batch 后 Loss 数值变了 | 按局部 token 数平均 | 小张量对比局部平均与全局平均 |

## 6. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| `fit()` 是单机代码 | 每个 `wg.` 调用都是一轮 Scatter → SPMD → Gather |
| `meta_info` 也按 batch 切 | 不切，每份原样给一份 |
| `DataProtoFuture` 能让同步 PPO 跳过 Driver | Driver 每步都要 union、算 Advantage，它几乎用不上 |
| 读最新 main 就能对上经典教程 | 经典文件已删除重构，要选对版本 |
| 权重同步要压到毫秒 | 秒级是常态，难点在正确性与峰值显存 |
| 推训 Logprob 必须逐位对齐 | 默认有偏差，用算法修正 |
| 读懂框架源码就掌握了后训练 Infra | 要有测量、有产出，最好亲手写一遍最小实现 |

## 7. 一分钟复习

1. Driver 持有整 batch；`wg.` 调用 = 切分 → 分发 → SPMD → 收集。
2. DataProto：Tensor 按第 0 维切；非 Tensor 共用边界；meta 不切。
3. union 全量比较、序列化整块拷贝：大 batch 时 Driver 是瓶颈。
4. 每个 Infra 设计都对应一个算法需求：共置、单控制器、推训切换、全局归一化、长度均衡、异步 Agent。
5. 读源码四步：算什么 → 什么约束 → 哪段代码 → 动手验证。
6. 校准：权重同步秒级；精度偏差靠算法修正；“怎么测量”是核心能力；亲手写 nano-GRPO 性价比最高。

## 8. 自测问题

### 问题 1

为什么说 veRL 是“Single-Controller + SPMD”的混合模型？哪一部分是单控制器，哪一部分是 SPMD？

期望回答：Driver 上的 `fit()` 持有全局控制流和整个 batch，按顺序调用各角色（单控制器）；每次调用被切分分发到多个 GPU Worker，Worker 内部的训练/推理引擎以 SPMD 方式并行执行，结果再收回 Driver。

### 问题 2

`fit()` 里为什么 `batch.repeat` 必须和生成时用同样的 interleave 方式？

期望回答：`union` 按行对齐两个 DataProto；生成侧是 `[p0, p0, …, p1, p1, …]` 的交错展开，原 batch 若按另一种顺序展开，行就对不上，uid 分组和 Advantage 都会错。

### 问题 3

一份能力清单说“权重同步要毫秒级完成”“推训 Logprob 必须对齐到完全一样”，你怎么评价？

期望回答：大模型权重同步实际是秒级，难点在 layout/dtype 正确与峰值显存；推训偏差默认存在，现在多用重要性采样一类方法在算法上修正，MoE 还要处理路由不一致。

## 9. 尚未解决与后续路线

- 阶段 2 留下的两道思考题：Driver 带宽估算；长度均衡与分发如何配合；
- 阶段 3～7 按计划逐个精读，尤其是推训切换的显存时间线与权重同步数据量；
- main 上新 Trainer（同步、共置异步、分离异步）的演进动机；
- 推训精度偏差的重要性采样修正（TIS/MIS）待系统学习。

## 10. 实验或参考材料

- veRL（HybridFlow）论文与 `volcengine/verl` v0.5.0 源码：`protocol.py`、`ray_trainer.py`、`single_controller/`、`fsdp_workers.py`、`sharding_manager/`。
- OpenRLHF；基于 Megatron + SGLang 的精简 RL 框架；全异步 RL 框架。
