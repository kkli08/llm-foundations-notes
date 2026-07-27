# 训练部署 Baseline 与 RLVR 单步闭环

> 日期：2026-07-27
> 来源：Codex Session `019f8e1a-1ff1-7002-aa46-85ffd51a11cc`；当日实际训练过程的脱敏理论抽取
> 状态：已整理

## 整理记录

- 2026-07-27 20:58：整理训练部署分层、Ray 集群、RLVR 单步、正确性证据和观测体系。
- 只保留可迁移的基础理论；内部地址、任务编号、队列、镜像、凭据、具体存储路径和工作安排均未写入。
- 本文承接 [2026-07-24 的训练与 RL 后训练基础](训练与RL后训练基础_20260724.md)，重点从“知道概念”前进到“能够运行、验收和定位一次 baseline”。
- Qwen3 MoE、参数分片和 MTP 适配接口另见 [Qwen3 MoE 与 MTP 适配最小架构](Qwen3MoE与MTP适配最小架构_20260727.md)。

---

## 0. 今天最终要解决的问题

1. 资源平台、Ray、训练编排层、Megatron 和 vLLM 分别负责什么？
2. 为什么资源任务显示 `RUNNING`，仍不能证明 Ray 集群或训练已经可用？
3. 什么是 smoke、baseline、正确性实验、效果实验和性能实验？
4. 一步 RLVR 训练究竟包含多少 Prompt、回答、Forward 和参数更新？
5. 数据集、Reward 和配置为什么也属于训练正确性的一部分？
6. 哪些日志可以证明模型真正执行了 Backward、Optimizer 和权重同步？
7. SwanLab、Grafana、平台日志和 Profiler 分别应该看什么？
8. 断开终端、退出 tmux、任务取消和日志持久化之间是什么关系？

---

## 1. 先给结论

### 1.1 一条完整训练部署链路

```text
本地电脑
→ 资源平台：申请机器、GPU、容器、镜像和存储
→ Ray Cluster：把多台机器组织成一个可调度集群
→ 训练编排层：创建 Actor、Reference、Rollout 等角色
├── Megatron：训练侧 Forward / Backward / Optimizer
└── vLLM：推理侧 Rollout / Prefill / Decode
→ 权重转换与在线同步
→ 日志、实验面板和系统监控
```

一句话记忆：

> 资源平台给机器，Ray 调进程，训练框架编排 RL，Megatron 负责训练，vLLM 负责生成。

### 1.2 这次 baseline 的准确定位

本次跑通的是：

```text
Qwen3 30B MoE
+ MTP-off
+ Offline Math Rule RLVR
+ 1 个外层训练 Step
+ 小规模合成数据
= 一次完整链路的 correctness smoke
```

它证明了完整链路可以执行，但没有证明：

- 多步训练能够收敛；
- 模型能力已经提高；
- 正式数据集可用；
- Checkpoint 保存和恢复正确；
- 稳态吞吐或 GPU 利用率达标；
- MTP 训练或投机解码已经生效。

### 1.3 最强的单步成功证据

```text
Rollout 生成回答
→ Reward 产生非恒定分数
→ Reference / Actor Logprob 完成
→ Advantage 非零且数值有限
→ Actor Backward / Optimizer 完成
→ Grad Norm 非零且有限
→ 新权重同步到 Rollout
→ Train Step 完成
→ 主程序正常退出
```

仅看到模型加载成功、Ray 正常或 vLLM 能生成文字，都还不够。

---

## 2. 先把系统分层：每层只回答一种问题

### 2.1 本地电脑：控制入口

本地电脑通常负责：

- 使用 CLI 或网页提交资源任务；
- 查看任务状态和平台事件；
- 进入远端 Head 容器；
- 编辑、同步或上传代码；
- 下载或查看持久化日志。

本地电脑通常不承担多节点 GPU 训练本身。训练命令从本地发出，不等于计算发生在本地。

### 2.2 资源平台：机器与容器生命周期

资源平台负责：

- 分配节点和 GPU；
- 拉取镜像、创建容器；
- 挂载共享存储；
- 注入普通环境变量和 Secret；
- 执行入口脚本；
- 维护任务的排队、初始化、运行、失败和取消状态。

它不理解 PPO、Actor、Reward 或 MTP。因此：

```text
平台任务 RUNNING
≠ Ray 已经 Ready
≠ 模型已加载
≠ 训练已经开始
≠ Baseline 已跑通
```

### 2.3 Ray：多节点进程调度

Ray 把多台独立机器组成一个逻辑集群：

```text
Ray Head
├── Worker Node 1
├── Worker Node 2
└── Worker Node 3
```

Head 负责：

- 接收 Driver 提交的任务；
- 维护节点和资源视图；
- 调度远程 Actor/Task；
- 追踪进程生命周期和失败。

Worker 负责：

- 接收调度；
- 启动训练、推理或辅助进程；
- 使用本机 CPU/GPU 做实际计算。

在 Head 提交训练的含义是“从统一入口调度整个集群”，不是“所有计算只在 Head 上完成”。

### 2.4 训练编排层：把算法步骤连成闭环

训练编排层通常负责：

- 读取 YAML 和命令行 Override；
- 构造数据集和 Workflow；
- 创建 Actor、Reference、Rollout 等角色；
- 组织 Rollout、Reward、Advantage、Update 和 Weight Sync；
- 记录模型版本和训练指标。

它通常不会重新实现矩阵乘或 Attention Kernel，而是把 Megatron、vLLM、Ray、Reward 和监控系统协调起来。

### 2.5 Megatron：分布式训练引擎

Megatron 主要负责：

- 分布式模型构建；
- Forward；
- 训练所需的 Logits/Logprob；
- Backward；
- Optimizer Step；
- TP、PP、EP、CP 等训练并行；
- 分布式 Checkpoint 或参数导出。

### 2.6 vLLM：高吞吐 Rollout 引擎

vLLM 主要负责：

- Prompt 的 Prefill；
- 自回归 Decode；
- KV Cache 管理；
- 动态/连续批处理；
- 生成 Response；
- 保存生成 Token 的行为策略 Logprob；
- 在启用时执行推测解码。

Megatron 和 vLLM 使用同一个逻辑模型，但面向不同目标：

```text
Megatron：高效训练，需要计算图、梯度和 Optimizer
vLLM：高效生成，需要 KV Cache、请求调度和吞吐
```

因此训练后必须把 Actor 新权重同步到 vLLM，不能只更新 Megatron 中的模型。

---

## 3. Head、Worker、Driver 与各种 Rank

### 3.1 Ray Head 不等于模型 Rank 0

Ray Head 是集群调度角色；训练 Rank 是某个分布式进程组内部的编号。

一个 Ray Cluster 中可以同时存在：

```text
Actor 训练进程组
├── Rank 0
├── Rank 1
└── ...

Reference 进程组
├── Rank 0
└── ...

vLLM 进程组
├── Rank 0
└── ...
```

每个进程组都可能有自己的 Rank 0。

### 3.2 三个常见编号

| 名称 | 回答的问题 |
|---|---|
| Node Rank | 当前机器在多节点任务中是第几个节点？ |
| Global Rank | 当前进程在整个分布式进程组中是第几个？ |
| Local Rank | 当前进程在本机上使用第几张 GPU？ |

它们可能在简单配置中碰巧相似，但不是同一个概念。

### 3.3 Driver 是什么

Driver 是发起 Ray 任务的主 Python 进程。它通常运行在 Head 上，负责创建远程进程和组织控制流。

```text
Head：机器/节点角色
Driver：当前任务的主控进程
Ray Actor：Ray 管理的有状态远程进程
RL Actor：被训练的策略模型角色
```

两个 `Actor` 含义不同，阅读日志和代码时必须结合上下文。

---

## 4. Baseline、Smoke、效果实验和性能实验

### 4.1 Baseline

Baseline 是后续变更的可比较起点。它应该：

- 配置明确；
- 能重复运行；
- 有清晰成功标准；
- 保存关键日志与版本；
- 尽量少引入新变量。

当前先跑 MTP-off，是为了建立“普通 RLVR 能运行”的参照。以后 MTP-on 出错时，才能判断问题是否由 MTP 引入。

### 4.2 Smoke Test

Smoke Test 是低成本的链路冒烟检查，目标是尽早发现：

- 配置无法解析；
- 依赖缺失；
- 路径不存在；
- Shape/Mask 不匹配；
- OOM；
- 分布式通信失败；
- Reward 没有有效信号；
- 权重没有同步。

Smoke 可以缩短：

- 训练步数；
- Prompt 数量；
- 每个 Prompt 的采样数；
- 最大生成长度；
- 数据集规模。

但不能随意改变模型并行布局、Checkpoint 格式和关键算法语义，否则会把“验证原链路”变成另一个未经验证的实验。

### 4.3 正确性、效果和性能是三个维度

| 实验类型 | 核心问题 | 主要证据 |
|---|---|---|
| 链路正确性 | 整个闭环能否无错完成？ | 阶段日志、数值健康、版本同步 |
| 模型效果 | 训练后能力是否提高？ | 独立验证集、Held-out Reward/Accuracy |
| 系统性能 | 资源是否高效利用？ | 稳态吞吐、时延、GPU/网络/显存指标 |

一轮训练完成只能证明正确性的一部分，不等于效果提升或性能达标。

---

## 5. 配置合成与数据契约

### 5.1 为什么正式运行前先检查最终配置

实际配置可能来自：

```text
基础 YAML
+ 环境变量插值
+ 命令行 Override
+ 框架默认值
= 最终运行配置
```

如果只读基础 YAML，可能误判真正生效的：

- Batch Size；
- 训练步数；
- 生成长度；
- 模型/数据路径；
- 日志上报模式；
- 并行配置。

因此运行前应验证“合成后的配置”，而不是只相信自己输入过哪些参数。

### 5.2 数据可加载不等于数据语义正确

一个数据集可能具备正确的 Parquet/JSON 格式，却仍然无法提供有效训练信号。

RLVR 数据通常至少需要：

```text
Prompt / Messages
Data Source 或任务类型
Ground Truth / Verifier 输入
必要的元数据
```

Reward 路由可能依赖 `data_source`。如果字段名称没有进入预期的 Reward 分支，最危险的情况不是立即报错，而是：

- Reward 恒为默认值；
- 评分规则完全未执行；
- 程序正常跑完但没有正确学习信号。

因此最小 Reward 单元测试应同时验证：

```text
已知正确回答 → 正分
已知错误回答 → 负分或较低分
```

### 5.3 合成数据能证明什么

合成小数据适合验证：

```text
数据加载
→ Rollout
→ Reward
→ Logprob
→ Advantage
→ Backward / Optimizer
→ Weight Sync
```

但它不能证明正式数据效果，也不能替代真实分布上的评估。

---

## 6. 一个外层 RLVR Step 到底发生了什么

### 6.1 完整控制流

```text
1. 从训练集取得一批 Prompt
2. Rollout 为每个 Prompt 生成若干 Response
3. Verifier / Reward 为每条 Response 打分
4. Reference 对相同 Response Token 计算 Logprob
5. Actor 对相同 Token 重算当前 Logprob
6. 计算 Advantage
7. 构造 Policy Loss 与约束项
8. Backward 计算梯度
9. Optimizer Step 更新 Actor 权重
10. 把新权重同步给 Rollout
11. 汇总并提交指标
```

### 6.2 一个 Step 不等于一条回答

假设：

```text
4 个 Prompt
× 每个 Prompt 生成 2 个回答
= 8 条 Trajectory
```

一个外层 Step 处理的是一批 Trajectory，而不是只生成一个 Token 或一条回答。

### 6.3 一个 Step 也不等于一次 GPU Forward

一批 Trajectory 还可能按照：

- Data Parallel；
- Microbatch；
- Token 上限；
- TP/PP/EP 布局；
- Reference/Actor 的不同阶段；

拆成多次 Forward 和 Backward。

因此：

```text
Global Step：算法/训练循环的外层单位
Microbatch：单次或小批 GPU 计算的数据单位
Kernel：GPU 上更底层的一次计算单位
```

三者不能混为一谈。

### 6.4 为什么 1-Step 还没有验证连续迭代

单步可能证明：

```text
Actor Version 0
→ Optimizer
→ Actor Version 1
→ Version 1 同步给 Rollout
→ 退出
```

它还没有证明：

```text
下一轮 Rollout 真正使用 Version 1
→ 再产生数据
→ Actor 更新为 Version 2
```

因此 2-Step smoke 的新增价值不是“训练时间更久”，而是验证权重版本能够跨轮次闭环。

---

## 7. 同一 Prompt 的多回答与组内 Advantage

### 7.1 什么是 Group

一个 Group 是同一个 Prompt 生成的多条回答：

```text
Prompt A
├── Response A1 → Reward 1
└── Response A2 → Reward 0
```

不同 Prompt 的回答不能直接混成同一个组。

### 7.2 为什么做组内比较

没有 Critic 时，可以用同一 Prompt 的平均表现作为简化 Baseline：

```text
回答得分 - 同组平均得分
```

高于同组平均的回答得到正 Advantage，低于同组平均的回答得到负 Advantage。

简化形式：

```text
Group Advantage
= (Reward - Group Mean)
  / (Group Std + epsilon)
```

它消除了部分 Prompt 难度差异，并把不同组的更新尺度拉到更接近的范围。

### 7.3 二元 Reward 的三种情况

```text
[1, 0] → 有正负相对信号
[1, 1] → 组内无法区分，任务相对信号接近 0
[0, 0] → 组内无法区分，任务相对信号接近 0
```

所以增加同一 Prompt 的采样数，可能提高组内出现好坏差异的概率，但会显著增加 Rollout 成本。

### 7.4 回答级 Reward 怎样影响 Token

回答级 Reward/Advantage 最终会通过 Mask 和 Return/Advantage 处理作用于 Response Token：

```text
正 Advantage 的 Response Token
→ 倾向于提高其概率

负 Advantage 的 Response Token
→ 倾向于降低其概率
```

Prompt Token 仍参与 Forward、提供上下文，但通常 `loss_mask=0`，不直接承担 Policy Loss。

---

## 8. 如何证明一次 Baseline 真正跑通

### 8.1 分层证据阶梯

| 层级 | 最小证据 | 仍未证明 |
|---|---|---|
| L0 资源 | 平台任务运行、节点/GPU 分配成功 | Ray、模型和训练 |
| L1 集群 | Ray 节点和资源完整，无 Pending/Failure | 依赖、路径和模型 |
| L2 初始化 | 依赖、数据、Tokenizer、Checkpoint、模型加载成功 | Rollout 和训练更新 |
| L3 生成 | Rollout、Reward、Logprob、Advantage 完成 | 参数真正更新和同步 |
| L4 单步闭环 | Backward、Optimizer、Weight Sync、Step 完成 | 连续迭代和效果提升 |
| L5 连续闭环 | 下一轮使用新版本继续 Rollout/Update | 收敛和泛化 |

### 8.2 单步日志应该找什么

最有信息量的证据是：

- Reward 不全相同；
- Advantage 非零、无 NaN/Inf；
- Actor Update 阶段实际执行；
- Grad Norm 有限且非零；
- 权重版本发生变化；
- Weight Sync 开始并完成；
- Step 完成；
- 程序正常退出。

### 8.3 `Grad Norm > 0` 能说明什么

它能支持：

- Backward 产生了非零训练信号；
- 训练不是完全空跑。

它不能单独证明：

- 每个预期参数都进入 Optimizer；
- 更新方向正确；
- 模型能力提高；
- 梯度没有被后续 Skip；
- MTP 参数被更新。

还需要联看 Optimizer、参数版本、同步和多步行为。

---

## 9. 如何区分“跑通、学会、跑快”

### 9.1 流程成功

关注：

- 所有阶段完成；
- 无异常；
- Tensor Shape、Mask、版本一致；
- 数值有限；
- 参数更新和同步发生。

### 9.2 模型效果

至少需要：

```text
多步训练
+ 可信训练数据
+ 独立验证集
+ 训练前后相同评测
+ Eval Reward / Accuracy 趋势
+ KL、Entropy、Grad Norm 等健康指标
```

训练 Batch 上 Reward 变高，不等于 Held-out 能力提高。

### 9.3 系统性能

至少需要：

```text
排除初始化阶段
→ Warm-up
→ 连续多个稳态 Step
→ 分阶段计时
→ 对齐 GPU、显存、网络和队列指标
```

需要拆开：

- Rollout 时间；
- Reference/Actor Forward；
- Backward/Optimizer；
- Weight Sync；
- 数据和调度等待。

单步总耗时不能告诉你真正瓶颈在哪。

---

## 10. 观测工具怎样分工

### 10.1 原始训练日志

最适合：

- 第一次异常；
- 控制流是否到达某阶段；
- Traceback；
- 模型版本和同步里程碑；
- 单次运行的精确上下文。

它是定位“哪里先坏了”的首要证据。

### 10.2 实验面板：SwanLab 一类系统

主要观察随 Step 变化的算法和训练指标：

- Reward；
- Advantage；
- Actor/Policy Loss；
- KL、Importance Weight、Clip Fraction；
- Entropy；
- Grad Norm；
- Rollout/Train/Sync 的时间曲线；
- 实验配置和代码版本。

只有同时满足以下条件，前端才会出现数据：

```text
在线上报开启
+ 凭据可读取
+ 集群网络可达
+ 训练至少提交一次指标
+ 没在首个 Commit 前失败
```

“账号已经配置”不等于“这次实验已经成功上报”。

### 10.3 Prometheus / Grafana

主要观察系统时间线：

- GPU 利用率和显存；
- 网络通信；
- 推理队列；
- KV Cache 使用；
- 请求吞吐和时延；
- 容器和服务健康。

“打开采集开关”只代表满足部分前提。完整接入还需要验证：

```text
Exporter/metrics endpoint 有数据
→ 平台完成抓取
→ Grafana 有权限和 Dashboard
→ 能按任务和时间范围筛选
```

### 10.4 资源平台

主要观察：

- 排队和调度；
- Pod/容器生命周期；
- 镜像拉取；
- Mount、OOM、Eviction；
- 平台级失败和退出码。

### 10.5 Profiler

Nsight、NCU 或 PyTorch Profiler 适合在已经定位到具体阶段后继续下钻：

```text
先用日志/指标定位慢在哪个阶段
→ 再用 Profiler 定位哪个算子或 Kernel
```

第一次 baseline 不应一开始就钻 Kernel。

---

## 11. tmux、临时文件与持久化存储

### 11.1 tmux 解决什么问题

tmux 保存远端终端会话和其中运行的进程：

```text
训练运行在 tmux
→ 本地网络断开或 Shell 退出
→ 远端 tmux 和训练进程仍可继续
```

它不能抵抗：

- 资源任务被取消；
- 容器被销毁；
- 节点故障；
- 任务达到最长运行时间。

### 11.2 `/tmp` 日志的边界

容器内 `/tmp` 通常只在容器生命周期内存在：

```text
断开 SSH/Shell → 通常仍在
任务取消/Pod 销毁 → 通常消失
```

因此取消资源前应把有价值日志复制到持久化存储。

### 11.3 环境变量不会自动跨 Shell

某个 Shell 中：

```bash
export RUN_LOG=/tmp/run.log
```

新开的 Shell 或 tmux 窗口不一定继承之后设置的变量。变量为空时：

```text
"${ARCHIVE_DIR}/head.log"
```

可能意外展开为：

```text
/head.log
```

因此执行归档、取消等有状态操作前，应重新检查关键变量的非空值和目标路径。

---

## 12. 用分层思维定位失败

| 现象 | 优先归属层 | 第一批检查 |
|---|---|---|
| 资源任务无法创建 | 资源平台 | 配额、权限、队列、镜像、参数 |
| 任务 RUNNING 但 Ray 不存在 | 入口脚本/Ray 初始化 | 进程、入口日志、Secret/依赖 |
| Ray 节点数不足 | Ray/网络 | Worker 加入、地址、端口、资源视图 |
| 四个节点看到的路径不同 | 存储/挂载 | 各节点只读 Path Check |
| 数据可读但 Reward 恒定 | 数据/Reward 语义 | Data Source 路由、正反例单测 |
| 模型加载失败 | Checkpoint/架构/并行 | Config、权重、Shape、分片 |
| Rollout 成功但 Actor Update 失败 | 训练侧 | Logprob、Mask、Advantage、Backward |
| Actor 更新但 Rollout 不变化 | 权重同步/版本 | 转换覆盖、版本号、第二轮 Rollout |
| 程序退出码正常但结果不可信 | 语义正确性 | Reward、Mask、版本、指标与验收标准 |

排查原则：

> 先找最早失败的层，再查该层输入；不要在资源层失败时修改模型超参数。

---

## 13. 安全与可复现性也是 Baseline 的一部分

### 13.1 Secret 不应进入普通参数或日志

凭据应通过平台的 Secret 机制注入，并只验证“是否存在”，不打印实际值。

需要防止凭据进入：

- 命令历史；
- 任务元数据；
- 日志；
- 聊天；
- Git；
- 实验名称和标签。

### 13.2 失败重试一次只改变一个主要变量

```text
固定其他条件
→ 只修复已确认的根因
→ 重新验证同一失败点
```

这样才能判断修复是否有效。一次同时改环境、数据、并行度和模型配置，会失去因果证据。

### 13.3 记录什么才可复现

至少记录：

- 代码版本；
- 脱敏后的最终配置；
- 模型和数据版本；
- 资源布局；
- 开始/结束时间；
- 成功标准；
- 关键里程碑和失败证据；
- 是否正式数据、是否 MTP-on；
- 是否完成清理和日志持久化。

---

## 14. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| 资源任务 RUNNING 就能训练 | 它只证明容器在运行；还要验证 Ray、依赖、路径和模型 |
| 在 Head 运行命令就只用 Head GPU | Head 是统一提交入口，Ray 会调度整个集群 |
| Ray Head 就是模型 Rank 0 | 一个是集群角色，一个是具体进程组编号 |
| 模型加载成功就是 baseline 跑通 | 还未验证 Rollout、Backward、Optimizer 和 Weight Sync |
| vLLM 能回答就证明训练成功 | 只能证明部分推理链路可用 |
| 一个训练 Step 就是一条回答 | 一个 Step 可包含多个 Prompt、多个回答和多次 Forward/Backward |
| Exit Code 0 就证明训练语义正确 | Reward、Mask 或数据路由仍可能静默错误 |
| Reward 平均值提高就证明能力提高 | 还需要独立验证集和多步趋势 |
| 打开 Prometheus 开关就等于 Grafana 接好了 | 还需验证 Endpoint、抓取、权限和 Dashboard |
| tmux 会永久保存训练和日志 | 它只能抵抗连接断开，不能抵抗容器销毁 |
| 合成数据跑通就是正式 baseline | 它只是 plumbing/correctness smoke |
| Grad Norm 非零就证明所有参数都正确更新 | 仍要检查 Optimizer、参数版本和同步覆盖 |

---

## 15. 一分钟复习

1. 资源平台给机器，Ray 管分布式进程，训练编排层组织 RL，Megatron 训练，vLLM 生成。
2. `JOB_RUNNING` 只代表容器运行；`ray status` 正常才代表集群形成。
3. Baseline 是可比较起点；Smoke 是低成本链路检查。
4. 一个 RL Step 包含一批 Prompt、多条回答、Reward、Logprob、Advantage、Update 和 Weight Sync。
5. 一步闭环最强证据是 Backward/Optimizer、版本变化、同步完成和 Step 正常结束。
6. 1-Step 没有验证下一轮 Rollout 真正使用新权重；2-Step 可以补这个证据。
7. SwanLab 看训练曲线，Grafana 看系统资源，原始日志看最早异常，Profiler 最后下钻 Kernel。
8. tmux 保存会话，不保存已经销毁的容器；临时日志必须在取消任务前复制到持久化存储。

---

## 16. 自测问题

### 问题 1

为什么平台显示 `RUNNING`，仍可能完全没有开始训练？

期望回答：平台只确认资源和容器生命周期；入口脚本、Ray、依赖、路径、模型和训练 Driver 都可能尚未 Ready 或已经失败。

### 问题 2

为什么在 Ray Head 上提交训练，不表示训练只使用 Head 节点？

期望回答：Head/Driver 是统一提交和调度入口，远程 Actor/Task 会被 Ray 放到整个集群的 Worker 上运行。

### 问题 3

为什么模型加载成功不能算 RLVR baseline 跑通？

期望回答：还没有证明 Rollout、Reward、Logprob、Advantage、Backward、Optimizer 和 Actor→Rollout 权重同步。

### 问题 4

4 个 Prompt、每个生成 2 个回答时，一个外层 Step 至少有几条 Trajectory？它是否只对应一次 Forward？

期望回答：8 条；不只一次 Forward，还会因角色、并行布局和 Microbatch 被拆成多次模型计算。

### 问题 5

为什么 `[1, 1]` 和 `[0, 0]` 的组内任务 Advantage 都可能接近 0？

期望回答：组内回答得分相同，没有相对优劣；减去组均值后都接近 0。

### 问题 6

1-Step 与 2-Step smoke 的核心证据差异是什么？

期望回答：1-Step 证明新权重可以产生并同步；2-Step 进一步证明下一轮 Rollout 真正使用新版本并继续闭环。

### 问题 7

SwanLab 和 Grafana 的主要分工是什么？

期望回答：SwanLab 主要看 Reward、Loss、KL、Gradient 等实验曲线；Grafana 主要看 GPU、显存、网络、队列、KV Cache 和吞吐等系统指标。

### 问题 8

为什么数据能够被 Dataset Loader 读取，仍可能是错误数据？

期望回答：字段格式正确不代表任务语义正确；Reward 路由、Ground Truth 或 Data Source 可能错误，导致评分分支未执行或 Reward 恒定。

---

## 17. 下一步学习路线

按当前任务优先级继续：

1. 闭卷画出一次 RL Step 的对象变化：Token、Logprob、Reward、Advantage、Version。
2. 用 2-Step MTP-off smoke 验证 Version 0 → 1 → 2 的连续闭环。
3. 验证一次在线实验指标上报和 Grafana 系统指标链路。
4. 进入 [Qwen3 MoE 与 MTP 适配最小架构](Qwen3MoE与MTP适配最小架构_20260727.md)，理解为什么 MTP 适配不只是增加一个 YAML 开关。
5. 当实际需要诊断 Ratio、Clip、KL 或 Entropy 时，回补 [PPO/GRPO 学习债务](../../../inbox/2026-07-27.md#1425明确延期专题ppo-与-grpo)。
