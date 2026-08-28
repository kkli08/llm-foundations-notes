# MTP 显存生命周期与 Allocator 性能归因

> 日期：2026-08-28
>
> 来源：近期 MTP K2/D1、K2/D2 训练 Trace 与 CUDA Memory Snapshot 复盘
>
> 状态：已整理

## 0. 这篇笔记解决什么问题

训练 Trace 中看到几秒级 GPU 空洞或很长的 CPU 区间时，不能直接把它叫作“计算慢”“通信慢”或“显存碎片”。要先回答：当时有哪些大 Tensor 仍然存活、谁正在申请新显存、PyTorch allocator 在做什么，以及 GPU 是在计算还是等待 Host 完成内存准备。

这篇笔记把 MTP 多深度训练中的几个概念连起来：

```text
MTP Forward
→ LM Head 输出 BF16 Logits
→ Cross-Entropy 生成 FP32 Softmax/中间状态
→ 为 Backward 保存必要 Tensor
→ 下一 Depth 继续 Forward
→ 多代大 Tensor 的生命周期发生重叠
→ allocator 可能扩展或映射新 Segment
→ Host 等待被表现为 GPU Bubble
```

## 1. 最重要的结论

1. Activation Checkpoint 只影响被包进 checkpoint 区域的模块；不能自动覆盖区域外的 LM Head 和 Cross-Entropy。
2. BF16 Logits 与 FP32 Softmax 的词表张量尺寸相差约两倍；后者若为 Backward 保存，生命周期会跨过后续 MTP Depth。
3. K2/D2 的显存峰值不能简单理解为“两套完整 Depth 张量相加”；必须按真实创建和释放时序逐项列活跃 Tensor。
4. Trace 中大块 `segment_map` 与长 `empty_strided` 相邻，说明 GPU Bubble 可能来自同步的 allocator 扩容或映射，而不是那个区间正在执行大 GEMM。
5. CUDA Memory Snapshot 能观察 PyTorch allocator 的 allocation、segment 和 stack，但不能单独区分底层 VMM 映射、allocator 锁竞争、隐式同步等所有机制。
6. Ring Buffer 只保留最后若干事件；长 Step 的早期事件被覆盖后，不能用尾部快照还原完整时间线。

## 2. 先分清四类对象

### 2.1 参数

模型长期持有的权重，例如 Backbone、MTP Layer、LM Head。训练时还会有梯度和 Optimizer State。

### 2.2 Activation

Forward 产生、Backward 可能需要的中间结果，例如 Hidden State、Norm 输出、Attention/MLP 中间值。

### 2.3 Logits 与 CE 中间状态

对于 Token 数为 \(N\)、词表大小为 \(V\) 的全词表输出：

\[
\text{Logits Shape}=[N,V]
\]

若 Logits 使用 BF16，每元素约 2 Byte：

\[
M_{\text{BF16 logits}}\approx N\times V\times 2
\]

CE 为数值稳定或 Backward 可能生成并保存 FP32 Softmax 类状态，每元素约 4 Byte：

\[
M_{\text{FP32 state}}\approx N\times V\times 4
\]

所以同一 Shape 下，FP32 状态约为 BF16 Logits 的两倍。

### 2.4 Allocator Segment

PyTorch allocator 通常不是每个 Tensor 都直接向驱动申请一块全新物理显存。它维护较大的 Segment，再从 Segment 切 Block 给 Tensor。

```text
Tensor 请求
→ allocator 找可复用 Block
├── 找到：直接分配
└── 找不到：扩展/映射 Segment
             → 可能触发较长 Host 操作或同步
```

因此：

```text
Tensor allocation 很慢
≠ Tensor 的数学算子很慢
```

## 3. Activation Recompute 到底保存了什么

Activation Recompute 的目的，是用额外计算换显存：Forward 时少保存 checkpoint 区域内部的 Activation，Backward 时再重跑这段 Forward。

关键不是“模型开启了 recompute”，而是 checkpoint 边界画在哪里。

一种实际常见的 MTP 路径是：

```text
checkpoint 区域：
MTP 输入投影
→ MTP Transformer/Core Layer
→ Final Norm

区域外：
→ LM Head / Output Projection
→ Cross-Entropy
```

这意味着：

- checkpoint 区域内部的中间 Activation 可以少存、Backward 时重算；
- 区域外的全词表 Logits、CE 状态仍可能真实创建并保存；
- 所以“开了 Activation Recompute”不等于消除了词表轴大 Tensor。

这与 Actor Logprob Recompute 仍是两个概念：

| 名称 | 重算什么 | 目的 |
|---|---|---|
| Actor Logprob Recompute | 用 Actor 再算已有 Response 的 Logprob | 构造 PPO/GRPO Loss |
| Activation Recompute | Backward 前重跑部分 Forward | 节省 Activation 显存 |

## 4. 用 D1/D2 时序理解 Tensor 生命周期

假设训练两个 MTP Depth，并且 CE 需要为 Backward 保存 FP32 状态。

### 4.1 Depth 1

```text
D1 MTP Core
→ D1 BF16 Logits
→ D1 CE
→ 保存 D1 FP32 Softmax/CE 状态供 Backward
```

D1 CE 结束后，D1 BF16 Logits 若不再需要可以释放；但 D1 FP32 CE 状态可能仍然存活。

### 4.2 Depth 2

```text
D1 FP32 CE 状态仍存活
→ D2 MTP Core
→ D2 BF16 Logits
→ D2 CE
→ 创建并保存 D2 FP32 CE 状态
```

在 D2 CE 的关键窗口，相关大 Tensor 更接近：

```text
D1 FP32 CE 状态
+ D2 BF16 Logits
+ D2 FP32 CE 状态
```

而不是：

```text
D1 BF16 Logits + D1 FP32 状态
+ D2 BF16 Logits + D2 FP32 状态
```

因为 D1 BF16 Logits 通常在 D2 Projection 返回前已经失去最后使用点。

### 4.3 一个 Shape 推导示例

若某配置推导出每份 BF16 Logits 约 3.789 GiB，则对应 FP32 状态约 7.578 GiB。D2 CE 窗口的大 Tensor 合计近似为：

\[
7.578+3.789+7.578=18.945\ \text{GiB}
\]

这个数是按公开 Shape 与 dtype 推导的教学示例，不应当被当作所有模型、所有 CE 实现的固定值。

真正的方法是：

```text
记录 Shape × dtype
→ 找创建时间
→ 找最后使用点
→ 画生命周期重叠
→ 再解释峰值与 allocator 行为
```

## 5. `empty_strided`、`segment_map` 与 GPU Bubble 怎么联系

### 5.1 `empty_strided` 不代表 CPU 在做张量数学计算

`empty_strided` 的语义主要是“按指定 Shape/Stride 准备一块 Tensor 存储”。如果 allocator 已有合适 Block，它可能很快；如果没有，就可能触发更重的显存准备。

### 5.2 `segment_map` 表示什么

Memory Snapshot 中的 `segment_map` 表示 allocator 正在为 Segment 映射新的显存范围。若一个大 Tensor 申请附近出现：

```text
长 CPU empty_strided
↔ 大块 segment_map
↔ GPU Timeline 空洞
```

较强的解释是：Host 正在同步完成 allocator 的 Segment 扩展/映射，GPU 没拿到下一批可执行工作。

### 5.3 能证明和不能证明的边界

这组证据可以支持：

- 卡顿与大 Tensor 分配及 allocator Segment 扩展高度相关；
- Bubble 不是由同一时间段内的 GEMM 持续计算直接造成；
- 多 Depth 生命周期重叠可能把分配推过某个 allocator 阈值。

但它不能单独证明底层究竟是：

- CUDA VMM page mapping 本身慢；
- allocator 全局锁或线程竞争；
- 显式/隐式 device synchronize；
- 驱动回收、页表更新或其他运行时机制。

所以准确措辞应是：

> 证据支持同步的大块 allocator 扩展/映射是 GPU Bubble 的直接邻接原因；更底层机制仍需定向观测。

## 6. 为什么不能一看到 Reserved 大就说显存碎片

几个常见量：

- Active：仍被活跃 Tensor 使用的显存；
- Reserved：allocator 从设备保留的总显存；
- Inactive/Reusable：保留但当前没有活跃 Tensor 使用、可能复用的 Block；
- Non-releasable：由于 Block/Segment 布局等原因暂时不能归还的部分。

若两个 Rank 的 Active、Reserved 和峰值很接近，没有明显异常的不可复用小块堆积，就不能宣称某个 Rank 存在严重独有碎片。

即使没有严重碎片，allocator 仍可能因为当下没有“足够大且合适”的连续可复用 Block 而扩展 Segment。因此：

```text
发生 segment_map
≠ 已证明严重碎片
```

## 7. Memory Snapshot 的观察边界

### 能看见

- PyTorch allocator 管理的 Tensor allocation/free；
- Segment map/unmap；
- Allocation Stack；
- 某时刻的 Active/Reserved 布局；
- 有限事件窗口内的时间顺序。

### 不一定看见

- NCCL 或自定义库绕开 PyTorch allocator 的直接 CUDA allocation；
- 完整 CUDA Driver/VMM 内部耗时原因；
- CPU 线程为何阻塞；
- Ring Buffer 被覆盖前的旧事件；
- 没有打入 NVTX 时准确的业务 Phase。

若训练 Step 很长，而事件缓冲只覆盖最后一小段，就必须写：

> 快照只证明尾部窗口的 allocator 行为，不能代表整个 Actor Update。

## 8. 推荐的归因流程

```text
1. Session Tracer / NVTX：定位慢在哪个业务 Phase
2. Torch Profiler / Perfetto：定位 CPU op、CUDA kernel 与 GPU Bubble
3. Tensor Shape × dtype：计算大 Tensor 的理论尺寸
4. Memory Snapshot：确认 allocation、生命周期和 segment 变化
5. 跨 Rank 对齐：区分单 Rank 异常与共同阈值
6. 只对未解释区间追加工具
   ├── CPU Sampling：线程/锁/Host 调度
   ├── Nsys：CPU-CUDA、同步和跨 Rank 时间线
   └── NCU：Kernel 的带宽与算力计数器
```

不要一开始同时打开所有 profiler。它们会改变时序、增加开销，并让“观测到的慢”混入观察者开销。

## 9. 下一步最有信息量的实验

当前最有信息量的对照不是继续猜，而是构造 matched D1/D2：

- 同一模型、输入、Token 数、并行拓扑和 warm-up；
- 只改变 Train Depth；
- 同时采集相同 Rank 的 Phase Trace 与 Memory Snapshot；
- 比较第一个新出现的大 allocation、Segment 扩展和 Bubble；
- 若 D1 无扩展、D2 稳定触发，则支持“生命周期重叠跨过 allocator 阈值”；
- 若 D1 也触发，说明问题不是 D2 独有，需要继续看 Shape、缓存暖态或执行顺序。

## 10. 常见误解纠正

| 容易误解为 | 更准确的理解 |
|---|---|
| 开 Activation Recompute 后没有大 Activation | 只减少 checkpoint 区域内的保存；区域外 Logits/CE 仍存在 |
| FP32 CE 状态和 BF16 Logits一样大 | Shape 相同但每元素字节数约为两倍 |
| D2 峰值就是两套 D1 峰值相加 | 要按真实最后使用点画生命周期 |
| `empty_strided` 很长说明 CPU 算得慢 | 它可能在等待大块存储分配/映射 |
| `segment_map` 等于显存碎片 | 它只说明扩展/映射；碎片要看 Block 布局和可复用性 |
| GPU 空洞说明 GPU Kernel 慢 | 空洞通常表示 GPU 没有可执行工作，可能在等 Host/同步 |
| 两 Rank 指标接近就已证明没有 allocator 问题 | 只能排弱“单 Rank 特有异常”，共同阈值仍可能存在 |
| Snapshot 能解释全部显存 | 它主要覆盖 PyTorch allocator 管理范围 |

## 11. 一分钟复习

```text
MTP 多 Depth 显存：
D1 CE 为 Backward 保存 FP32 状态
→ D2 继续创建 BF16 Logits 和 FP32 CE 状态
→ 大 Tensor 生命周期重叠
→ allocator 找不到合适 Block 时扩展/映射 Segment
→ Host 卡住，GPU Timeline 出现 Bubble
```

```text
正确归因：
Phase 时间线
→ Tensor Shape/dtype
→ 创建与最后使用点
→ Memory Snapshot segment/block
→ 跨 Rank 对照
→ 再决定 CPU Sampling、Nsys 或 NCU
```

## 12. 自测问题

1. 为什么 Activation Recompute 不能自动消除 MTP 的全词表 Logits 和 CE 状态？
2. 为什么 D2 CE 时的活跃大 Tensor 不一定包含 D1 BF16 Logits？
3. `empty_strided`、`segment_map` 和 GPU Bubble 同时出现时，最谨慎的结论是什么？
4. 为什么 `reserved - active` 很大仍不足以单独证明严重碎片？
5. Ring Buffer 只覆盖长 Step 尾部时，结论必须怎样限定？
6. 下一次 D1/D2 matched Snapshot 要固定哪些变量？

## 13. 与已有知识的联系

- [Online MTP 训练成本与性能归因](OnlineMTP训练成本与性能归因_20260826.md)：先建立 K/D、CE Fusion、Recompute 和工具分工；本文进一步补齐真实 Tensor 生命周期与 allocator 证据链。
- [MTP 与 Context Parallel 训练正确性](MTP与ContextParallel训练正确性_20260817.md)：CP 切序列但不切词表，解释了为何词表轴大 Tensor 仍可能成为显存瓶颈。
- [On-policy 训推与 MTP 性能分析](OnPolicy训推与MTP性能分析_20260806.md)：补充了正式 Benchmark 与 Profiler 必须分离的原因。
