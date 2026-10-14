# 推理引擎的 KV 分页、连续批处理与 CUDA Graph

> 日期：2026-10-14
> 来源：[当日 Inbox](../../../inbox/2026-10-14.md)；只保留 vLLM V1 一类推理引擎的通用机制与 RL Rollout 下的含义，不记录内部代码路径与配置文件。
> 状态：已整理

## 整理记录

- 2026-10-14 18:52：Claude Code 自动整理（当天主整理）；来源 Claude Code 会话 `4777f0df`、`2e0044da`、`933963c1`（14:33–14:59）。

> 证据边界：讲解时本机没有安装推理引擎，上游内部模块名与行为按 V1 架构的公开设计讲解，没有对着具体小版本逐行核对；下文数字是为好算而设的数量级估计，以启动日志与 `/metrics` 实测为准。

## 0. 今天最终要解决的问题

用户想系统梳理推理栈，三个问题几乎同时提出：

1. PagedAttention 与 KV Cache 显存管理具体是怎么做的？
2. Continuous Batching 是怎么做的？
3. CUDA Graph 是什么？

三者是同一个推理引擎的三层：**KV 分页管显存，连续批处理管调度，CUDA Graph 管 CPU 发射开销**。

## 1. 先给结论

- **KV Cache 是推理时唯一随请求数与长度变化的大块显存**。PagedAttention 把它切成固定大小的 block，每个请求一张 block table，attention kernel 按表读不连续的显存——解决**碎片**和**共享**两件事。
- **CPU 只管簿记，GPU 只认编号**：哪个 block 属于谁、引用计数、哈希都在 CPU 侧；GPU 侧只有 block id 与 slot 编号，两边的契约就是 block id 列表。
- **前缀缓存的哈希里没有权重版本**：RL 换权重后必须清缓存，否则新请求会命中旧权重算出的 KV，Logprob 悄悄出错。
- **Continuous Batching = 每一步 forward 都重新决定算哪些 token**；V1 不再区分 prefill/decode，只问“还差多少 token 没进 KV”。它只有在 KV 能按块随时分配释放时才做得成。
- **batch 永远不会“跑完”**：权重更新没有天然停顿点，要人为造边界（暂停准入 → abort 在途请求 → 清缓存 → 客户端带着已生成部分续写）。
- **CUDA Graph 用“固定地址、固定 shape、固定路径”换“一次 launch 跑完整步”**：主要收益在 decode；RL 权重更新必须原地写入，否则图静默读旧地址。

## 2. 前置概念

### 2.1 每个 token 的 KV 字节数

```text
每 token KV 字节 = 2（K 和 V）× 层数 × KV heads × head_dim × dtype 字节
```

贯穿例子（8B 量级、GQA 8 个 KV head、36 层、head_dim 128、bf16）：

```text
2 × 36 × 8 × 128 × 2 = 147,456 字节 ≈ 144 KiB / token
block_size = 16 → 每块约 2.25 MiB
```

### 2.2 启动时决定池子多大

```text
KV 预算 = 总显存 × gpu_memory_utilization
        − 模型权重 − profile 时的峰值激活 − 非 torch 显存（NCCL、CUDA Graph 等）
num_blocks = KV 预算 / 每块字节数
```

80 GB 卡、利用率 0.9、权重约 16 GB、再扣几 GB 激活与图：约 50 GB 给 KV → 约 35 万 token、约 2.2 万个 block。

立刻得到一个推论：最大长度 32K 时，**每条都跑满只能同时容纳十条左右**。`max_num_seqs` 只是上限，**真正限制并发的是 KV 池**（与 [上下文越长为什么越慢](../07/基础理论_20260723.md#第九部分为什么上下文越长越慢) 的并发下降一致）。

可调的几个旋钮：`block_size`、`gpu_memory_utilization`、直接指定 KV 字节数（跳过推算）、KV dtype（如 fp8 让 token 容量约翻倍）、是否开前缀缓存。

## 3. 完整原理

### 3.1 为什么 RL Rollout 特别需要分页

- **长度方差大**：同一道题，有的采样 200 token 结束，有的跑到 8K。连续预分配要按最长占位，大部分是空的，并发被压低；
- **同题多采样**：GRPO 一个 prompt 采 n 条，n 份相同的 prompt KV 各存一份就是 n 倍浪费——分页 + 引用计数正好对症。

### 3.2 全局分工

```text
启动期：profile → KV 预算 → num_blocks → 按层分配 KV 张量

EngineCore 进程（CPU 簿记）
  Scheduler：每步决定谁跑、跑多少 token
    ↕ allocate_slots / free
  KVCacheManager → BlockPool（空闲队列 + 前缀哈希表 + 引用计数）

GPU Worker（数据面）
  ModelRunner：构建 block_table / slot_mapping
  Attention backend：按 slot 写 KV，按 block_table 读 KV
```

### 3.3 BlockPool 的三样东西

| 结构 | 作用 |
|---|---|
| 每块元数据 `{block_id, ref_cnt, block_hash}` | 物理块是谁的、被几个请求共享 |
| 空闲队列（双向链表，LRU 顺序） | `ref_cnt == 0` 的块；分配时从队头取，最久没用的先被淘汰 |
| `hash → block` 表 | 前缀缓存索引，**只收录写满的块** |

块哈希是**链式**的：`hash(父块哈希, 本块 token ids, 额外键)`，额外键包括多模态内容哈希、LoRA id、cache salt 等。所以同一个块哈希隐含“从开头到这里的全部前缀都相同”。

**重要推论：哈希里没有模型权重版本。**

块的状态机，关键是“空闲但仍挂着哈希”这个中间态：

```text
Free ──分配（从队头取；若带旧哈希就从哈希表移除）──▶ InUse
InUse ──前缀命中──▶ InUse（ref_cnt++）
InUse ──请求结束，ref_cnt 归零──▶ CachedFree（保留哈希，追加到队尾）
CachedFree ──新请求前缀命中（从空闲队列摘出）──▶ InUse
CachedFree ──被淘汰、重新分配给别人（清哈希）──▶ InUse
CachedFree ──清前缀缓存──▶ Free
```

请求释放时，块按**逆序**放回队尾：尾部块最不可能被别人共享，逆序放回让它们先被淘汰。

### 3.4 一个请求的生命周期（GRPO，prompt 1000 token，n = 8）

1. **查前缀**：按链式哈希逐块查表。1000 = 62 × 16 + 8，最多命中 62 个满块；剩下 8 个 token 不足一块，没有哈希，必须重算；而且**至少要算一个 token 才能拿到 logits**；
2. **分配**：`allocate_slots(新 token 数)` 检查空闲块是否够，不够返回失败（见 §3.5）；够就给命中块 `ref_cnt++`，再从队头取新块；
3. **共享**：第 1 个样本算完整个 prompt，之后的样本命中，那 62 块 `ref_cnt = 8`，每个样本只额外占自己的尾块——prompt KV 从 8 份降到约 1 份（8 个请求几乎同时到达时，能否在同一步互相命中取决于调度时序，应用前缀缓存命中指标实测）；
4. **GPU 执行**：把每个请求的 block id 填进 `block_table[num_reqs, max_blocks]`，为每个新 token 算 `slot = block_id × block_size + offset`；写：按 slot_mapping 把新 K/V 散写进池子；读：支持 block table 的 FlashAttention / FlashInfer 按表读历史 KV（老的手写 paged attention kernel 现在基本是遗留实现）；
5. **decode**：每生成 16 个 token 需要一个新块；块写满后算哈希登记，所以多轮 Agent 的下一轮能命中上一轮的全部历史；
6. **结束**：`free`，各块 `ref_cnt − 1`，归零的进入队尾，哈希保留，等复用或淘汰。

`block_size` 的取舍：块越小，尾块浪费越少、共享粒度越细；块越大，页表越短、kernel 效率越高；某些 attention backend 对块大小有约束。

### 3.5 显存不够：抢占

`allocate_slots` 失败时，Scheduler 抢占 running 队列里优先级最低的请求（FCFS 下是最后加入的那个）：

1. 释放它的全部块；
2. 把 `num_computed_tokens` 重置为 0；
3. 放回 waiting 队列，之后**重算**（V1 只做 recompute，不再 swap 到 CPU）。

重算不一定从头算：被释放的满块进入“空闲但挂着哈希”状态，只要还没被淘汰，再次接纳时查前缀就能命中，代价小很多。

### 3.6 Continuous Batching：每一步重新组 batch

| | 静态 batching | Continuous batching（iteration 级调度） |
|---|---|---|
| 调度粒度 | 一整批请求 | 每一步 forward |
| 短请求先结束 | 位置空着，等最长那条 | 当步释放，下一步补新请求 |
| 新请求到达 | 等下一批 | 下一步就可能接纳 |
| prefill 与 decode | 分阶段 | 同一步混在一起算 |
| 显存 | 按最大长度预留 | 按 block 按需分配 |

两件事同时成立才做得成：**调度器每步重新挑请求**（Orca 的思想）；**KV 可以按块随时分配和释放**（PagedAttention），否则请求随时进出会造成严重碎片。

**分工**：Scheduler 在 EngineCore 进程里，是核心；ModelRunner 只“给什么算什么”；每个请求维护 `num_computed_tokens`（已写进 KV）与 `num_tokens`（prompt + 已生成）。客户端只管并发发请求，batch 完全在引擎内部形成——这是异步 Rollout 设计能很简单的原因。

**V1 的关键简化**：不区分 prefill 阶段与 decode 阶段，对每个请求只问“还差多少 token 没算进 KV”，即 `num_tokens − num_computed_tokens`。decode 中的请求差 1；新请求差“prompt 长度 − 前缀命中”。每步有两个上限：token 预算与最大并发请求数。

例子（预算 2048；B、C 在 decode；新请求 A 的 prompt 3000，未命中前缀）：

| 步 | 调度 | 本步 token |
|---|---|---|
| t | 先排 running：B=1、C=1，剩 2046；再从 waiting 接纳 A，只给 2046（chunked prefill） | 2048 |
| t+1 | B=1、C=1、A 剩余 954；A 的 prefill 完成，这一步对 A 的采样才有效（中间 chunk 的采样被丢弃）；C 采到 EOS | 956 |
| t+2 | C 结束、释放 block；A=1、B=1，接纳 waiting 里的 D 做 prefill… | 2 + D |

状态流转：`WAITING →（预算够且 KV 能分配）→ RUNNING →（EOS/stop/max_tokens/abort）→ FINISHED → 释放 block`；KV 不足被抢占时回到 WAITING 队首。

### 3.7 每一步循环与“混合 batch 为什么正确”

一步 = `schedule → execute_model → update_from_output`：

1. 先遍历 running，`num_new_tokens = min(差值, 剩余预算)`，`allocate_slots`，失败就抢占；
2. 再遍历 waiting，查前缀缓存，同样 `allocate_slots`；
3. 产出调度结果：新请求带完整数据，老请求**只发增量**（新 block id、本步 token 数），worker 侧维护持久的 batch 状态；
4. ModelRunner 把所有请求的 token **拼成一条一维序列、不做 padding**，配上 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping`，varlen attention 在同一个 kernel 里同时处理多 token 的 prefill 与单 token 的 decode；
5. 采样结果追加到请求，检查停止条件，结束的释放 block。

**为什么混合 batch 不影响正确性**：attention 按请求分段算（每个请求只看自己的 KV block），MLP 与投影层对 token 逐行独立。所以把不同请求、不同阶段的 token 拼在一起只影响 kernel 效率。

相关旋钮：最大并发、每步 token 预算（即 chunk 大小）、chunked prefill（V1 生成任务强制开启）、前缀缓存、显存利用率与 block_size（池子不够就抢占）、异步调度（第 N+1 步调度与第 N 步 GPU 执行重叠）。

### 3.8 CUDA Graph：固定地址、固定 shape、固定路径

**解决什么**：GPU 是异步执行的，CPU 一个个发射 kernel。decode 时每请求只算 1 个 token，单个 kernel 只跑几到几十微秒、偏显存带宽瓶颈；一步 forward 却要发射几百到上千个 kernel（每层 norm、QKV、RoPE、attention、MLP、残差 × 层数），每次 launch 加上 dispatcher 与 Python 开销要十几到几十微秒——CPU 跟不上，GPU 时间线到处是空隙。prefill 每个 kernel 本身很重，这个问题主要在 decode。

**三阶段**：

1. Capture：在特殊 stream 上跑一遍 forward，kernel 只被记录不执行；
2. Instantiate：编译成可执行对象；
3. Replay：每步一次 `graph.replay()`，GPU 连续执行，中间不经过 CPU、Python 与框架。

```python
static_input = torch.zeros(bs, hidden, device="cuda")
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):          # 之前先 warmup 几次
    static_output = model(static_input)

static_input.copy_(new_input)      # 写进录制时那块显存
g.replay()
result = static_output.clone()     # 从固定输出 buffer 读
```

**核心约束：录下的是具体地址、具体 shape、具体执行路径**：

| 约束 | 后果 | 引擎怎么应对 |
|---|---|---|
| 地址固定 | 换一个 tensor 传进去图看不到 | static input buffer，每步 `copy_` |
| shape 固定 | 每张图对应一个 batch size | 为一组尺寸各录一张，运行时 pad 到最近的捕获尺寸 |
| CPU 逻辑不进图 | Python `if/for` 在录制时被固化 | 录制区域内不能有数据相关控制流 |
| 不能 CPU-GPU 同步 | `.item()`、`.cpu()` 录制时报错 | 采样等放图外或改写成纯 GPU 操作 |
| 不能录制中动态分配 | 图有专属 memory pool，录完一直占着 | 启动多花时间和显存 |

**Attention 最麻烦**：decode 时每个请求的长度和 block table 每步都在变，图要求元数据在固定地址、固定 shape，所以 backend 要做成图兼容形式（元数据先写进预分配 buffer，kernel 从 buffer 读）。vLLM V1 默认结合 torch.compile 用 **piecewise CUDA Graph**：attention 留在图外 eager 执行，其余切成几段分别录，prefill 与混合 batch 也能受益；FULL 模式把 attention 也录进去，前提是 backend 支持。SGLang 主要对 decode 做整图录制，batch 超过最大录制尺寸时回退 eager。

**常见权衡**：录的尺寸越多越大，覆盖越好，但启动越慢，图的 memory pool 还会挤占 KV 显存（§2.2 的预算公式里那一项）。

### 3.9 RL 与这三层的耦合

| 耦合点 | 要守住的规则 | 违反时会怎样 |
|---|---|---|
| 前缀缓存 × 换权重 | 换权重必须清前缀缓存（哈希里没有权重版本） | 新请求命中旧权重的 KV，Logprob 悄悄出错，数据变成 off-policy |
| 共享 × 回收 | `ref > 0` 的块绝不能被回收 | 共享前缀被覆盖，其他样本输出乱码 |
| 按 slot 索引的旁路缓冲 | 容量要覆盖整个池子（`num_blocks × block_size`） | 例如按 slot 记录 MoE 路由时越界或错位 |
| 抢占 | 被抢占请求的进度必须清零 | 重算时读到已被别人覆盖的块 |
| CUDA Graph × 权重更新 | 权重原地 `copy_`，地址不变；sleep/wake 后保证地址不变或重录 | 图静默使用旧权重或读到垃圾，不报错 |
| 连续批处理 × 换权重 | 人为造边界：暂停准入 → abort（带清缓存）→ 换权重 → 恢复 | 同一请求前后半段用不同权重却无记录 |

换权重时的典型流程：服务端先关准入栅栏，再以“abort + 清缓存”暂停生成；客户端收到 abort 后把已生成 token 拼到输入后面，等暂停结束重新提交，并逐 token 记录生成时的权重版本供训练侧处理 off-policy。代价是缓存刚被清空，**续写时“prompt + 已生成部分”要在新权重下整段重新 prefill**。

sleep/wake（训推共卡时）：sleep 把权重卸到 CPU、KV 直接丢弃，wake 时重建。

MTP / 推测解码与 CUDA Graph：verify 步每请求处理 `k+1` 个 token，图的 token 维是 `bs × (k+1)`；draft 每一步通常也单独录图；部分接受导致的长度不齐靠 padding 与固定 buffer 处理——这是 MTP + CUDA Graph 最容易出 bug 的地方（参见 [CUDA Graph、Kernel、Bucket 与 Profiler 的分工](../08/OnPolicy训推与MTP性能分析_20260806.md#103-cuda-graphkernelbucket-与-profiler-的分工)）。

## 4. 与已有知识的联系

- 与 [PagedAttention 是什么](../07/基础理论_20260723.md#32-pagedattention-是什么) 相关：那里讲了逻辑块到物理块的映射与“不改变 Attention 数学”；本篇补上块池簿记、链式哈希、引用计数、状态机、抢占与启动期容量计算。
- 与 [Prefix Caching 与 PagedAttention 不同](../07/训练与RL后训练基础_20260724.md#86-prefix-caching-与-pagedattention-不同) 相关：本篇说明二者在实现上如何结合——前缀缓存就是“满块哈希表 + 空闲但挂着哈希的块”。
- 与 [Rollout 瓶颈下的 GPU 利用率](Rollout瓶颈下的GPU利用率与对照变量_20261009.md) 中的 KV 抢占相关：抢占的真实代价是释放 + 进度清零 + 重算。
- 与 [配置三态与 Prefix Cache 风险](../09/配置三态与PrefixCache风险证据_20260907.md) 相关：本篇给出“换权重必须清缓存”的机制原因。
- 与 [PPO 为什么要重算 Logprob](PPO的Critic与Clip在单次更新下的退化_20261013.md#34-为什么要重算-logprob) 相关：CUDA Graph、前缀缓存、推测解码都是训推数值不同的来源。
- 本次填补了索引“待深入专题”中的 Continuous Batching。

## 5. 工程中的具体表现

| 现象 | 原因 | 如何验证 |
|---|---|---|
| 并发远低于最大并发配置 | KV 池按长度只能放下十几条 | 启动日志的 GPU block 数 × block_size ÷ 典型长度 |
| GRPO 同题采样显存远小于 n 倍 | 前缀满块被引用计数共享 | 前缀缓存命中率指标 |
| 长尾请求突然变慢 | 被抢占后重算 | 抢占计数；前缀命中能否接住 |
| 换权重后首轮 Logprob 偏差异常 | 没清前缀缓存，命中旧 KV | 换权重后比较开/关前缀缓存的 Logprob |
| 开图后结果错或 hang，eager 正常 | 地址变化、padding、元数据 buffer 未更新 | 切 eager 对比；权重更新是否原地 |
| Profiler 里单 kernel 归因困难 | 图整体执行 | 单独跑 eager 版本或用 nsys 图节点追踪 |

## 6. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| 最大并发配置决定能并发多少请求 | 真正的上限常是 KV 池容量 |
| 前缀缓存能识别权重是否变了 | 链式哈希只含 token、多模态、LoRA 等，没有权重版本 |
| 请求结束 block 立即清空 | 进入“空闲但挂着哈希”，可被复用，直到被淘汰 |
| 被抢占的请求一定从头算 | 进度清零，但满块若未淘汰可被前缀命中 |
| Continuous Batching 是把请求凑成更大的批 | 是每一步重新决定算哪些 token |
| prefill 和 decode 必须分开跑 | V1 统一为“还差多少 token”，混在同一步、一维拼接 |
| CUDA Graph 加速的是 GPU 计算 | 省的是 CPU 发射开销，主要在 decode |
| 换权重只要把新张量赋给参数 | 必须原地写入，否则图仍读旧地址 |

## 7. 一分钟复习

1. KV 池 = 显存 × 利用率 − 权重 − 激活 − 非 torch；每 token = 2 × 层 × KV heads × head_dim × 字节。
2. CPU 簿记、GPU 认编号；契约是 block id 列表。
3. 链式哈希只收满块、没有权重版本 → 换权重清缓存。
4. 空闲但挂着哈希 = 前缀复用；逆序放回让尾块先被淘汰。
5. 抢占 = 释放 + 进度清零 + 重算，前缀命中可接住。
6. 连续批处理：每步重排，只问“还差几个 token”；一维拼接无 padding。
7. batch 不会跑完：换权重要暂停、abort、清缓存、续写（整段重 prefill）。
8. CUDA Graph：地址、shape、路径都固定；分档 padding；权重原地更新；出问题先 eager。

## 8. 自测问题

### 问题 1

GRPO n = 8、prompt 1000 token、每个样本生成 2000 token、block_size 16，开前缀缓存时总共占多少个 block？关掉又是多少？

期望回答（自算，供核对）：共享 62 个满块；每个样本剩 8 个 prompt token + 2000 生成 = 2008 token，需 ⌈2008/16⌉ = 126 块；总计 62 + 8 × 126 = 1070 块。关掉时每个样本 3000 token 需 ⌈3000/16⌉ = 188 块，共 1504 块。

### 问题 2

换权重前已经 abort 了所有请求，如果不清前缀缓存，还会出问题吗？

期望回答：会。abort 后块进入“空闲但挂着哈希”状态，新权重下的新请求查前缀时会命中这些旧权重算出的 KV，Logprob 悄悄出错。

### 问题 3

被抢占的请求重算时一定要从头算吗？

期望回答：计算进度清零，但它写满的块进入空闲队列且保留哈希；若在被重新接纳前没被淘汰，查前缀即可命中，只需重算没命中的尾部。

### 问题 4

为什么把不同请求的 prefill token 和 decode token 拼成一条序列算不会出错？

期望回答：attention 按 `query_start_loc` 分段、每段只读自己的 block table；MLP 与投影层对 token 逐行独立。拼接只影响效率。

### 问题 5

RL 权重同步后 Rollout 静默用了旧权重，且开 eager 时正常。最可能的原因？

期望回答：权重更新换了新的 tensor 对象而不是原地 `copy_`，CUDA Graph 仍按录制时的地址读旧权重。

## 9. 尚未解决与后续路线

- 同一 prompt 的 n 个请求几乎同时到达时，能否同一步互相命中前缀，需要用命中指标实测；
- piecewise 图与 torch.compile 的具体切分、attention 元数据在图里如何更新；
- MTP verify 步的图 shape 设计；
- 多轮 Agent 等待工具返回时，KV 是挂起、卸载还是释放重算，各自代价。

## 10. 实验或参考材料

- vLLM 论文（PagedAttention）与 V1 架构文档：Scheduler、KVCacheManager、BlockPool、前缀缓存；
- Orca：iteration-level scheduling；
- PyTorch `torch.cuda.graph` 文档与 CUDA Graph 编程指南；
- FlashInfer 的 CUDA Graph 兼容 decode wrapper。
