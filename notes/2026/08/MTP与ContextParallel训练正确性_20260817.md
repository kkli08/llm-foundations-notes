# MTP 与 Context Parallel 训练正确性

> 日期：2026-08-17
> 来源：Codex Session `019ff0f2-6a27-7220-9da5-532f3f08bea2`、`01a00f54-c3d5-7cf3-80d6-7190d7166857`
> 状态：已整理

## 0. 这篇笔记最终要解决的问题

MTP 单卡或 `CP=1` 能训练，不代表 `CP>1` 只需改一个并行配置。Context Parallel 切开了序列，而 MTP 的监督和输入依赖未来 Token；跨 Rank 边界、Packed Segment、Loss 归一化、Recompute 和显存维度都必须同时正确。

## 1. 先给结论

1. CP 沿序列维度切 Token；MTP 的 `t+n` Label/Embedding 可能落在另一个 CP Rank，不能在本地分片上直接循环 Shift。
2. 正确实现必须同步处理 Future Label、Future Embedding、Loss Mask、Packed Boundary 和每个 MTP Depth 的 Padding Mask。
3. 全局 MTP Loss 应按 DP×CP 范围内的有效 Token `numerator / count` 归一化；某个 Rank 本地零有效 Token 是合法边界情况。
4. “逻辑预测深度 K”与“物理 MTP 层数”不是同一个量；一个物理层可以被递归复用多次形成 K2/K3。
5. CP 只切序列，不自动切词表维度；因此 CP 正确性跑通不等于任意长上下文都不会 OOM。
6. 目前的线性 Draft Chain 与 Tree Training/树状 Draft 是不同能力，不应混用同一组参数或验收标准。

## 2. 逻辑预测深度与物理 MTP 层数

以当前位置 $t$ 为例：

```text
NTP 主路径：预测 x(t+1)
MTP Depth 1：预测 x(t+2)
MTP Depth 2：预测 x(t+3)
```

这里的 Depth/K 表示逻辑上向未来预测多少步。物理实现可能是：

```text
一个物理 MTP Layer
→ 第一次使用，产生 Depth 1
→ 将预测 Token/Hidden State 作为下一次输入
→ 第二次复用同一组参数，产生 Depth 2
```

因此：

```text
logical prediction depth = 2
physical MTP layers = 1
```

完全可以同时成立。Checkpoint 记录的是实际存在的参数层；运行配置还要记录递归使用次数。把二者都叫 `num_layers` 容易造成加载、验证和日志语义混乱。

### 2.1 MTP 配置应拆成哪些语义

一套清晰的实现通常至少要区分：

| 配置语义 | 回答的问题 |
|---|---|
| `enable` | 是否构建/加载 MTP 模块 |
| `num_layers` | Checkpoint 中有多少物理 MTP 参数层 |
| `enable_train` | Actor Update 是否计算 MTP Loss |
| `prediction_depth` | 训练/推理逻辑上向未来展开多少步 K |
| `detach_encoder` | MTP Loss 是否回传到 Backbone/共享参数 |
| `mtp_loss_scaling_factor` | MTP 辅助目标在 Total Loss 中的权重 |
| Rollout `method=mtp` | 推理端是否启用 MTP Proposer |
| Rollout speculative tokens | 每轮希望 Draft 多少个 Token，通常与逻辑 K 对齐 |

这些开关属于不同契约：

```text
构建/加载 MTP
≠ 训练 MTP Loss
≠ Rollout 开启 Speculative Decoding
```

CP>1 是 Actor 训练拓扑能力，不应再发明一组 CP 专属 MTP 参数。它应让原有 MTP 语义在序列分片后保持正确，并在底层能力缺失时 Fail Fast。

### 2.2 多深度 Loss、共享物理层与 Teacher Forcing

K2 训练会产生两个逻辑 Loss：

```text
Depth 1 → mtp_1_loss
Depth 2 → mtp_2_loss
```

它们与 CP Size 无关。对 repeated-layer 实现，同一份物理参数 $W$ 被调用两次，两条计算路径的梯度在 Backward 时累加到同一个 $W$ 上，Optimizer 仍只更新一份物理层。

一种常见加权方式是：

$$
L_{\text{MTP}}
=
\frac{\alpha}{K}
\sum_{k=1}^{K}L_k
$$

例如 $\alpha=0.1,K=2$，则进入 Total Loss 的是 `0.05 * L1 + 0.05 * L2`。日志里单独展示的 `mtp_1_loss` 和 `mtp_2_loss` 通常是加权前的 Raw Loss，不应直接把它们当成对 Total Loss 的实际贡献。

训练时通常使用 Teacher Forcing：Depth 2 的 Future Token Embedding 来自 Ground Truth，而不是 Depth 1 刚采样出的 Draft Token。所以：

- Depth 1 预测错不会把一个错 Token 直接喂给 Depth 2；
- Depth 2 仍可以使用 Depth 1 的 Hidden State，梯度也可以穿过重复调用；
- 推理侧却是线性 Draft Chain，第一个 Draft 错误往往会让后续候选无法被 Target 接受。

`detach_encoder=true` 另外决定 MTP Loss 是否回传到 Backbone：它不改变 Teacher Forcing，也不改变同一 MTP 物理层被多次调用的事实。

## 3. CP 为什么会破坏朴素的 MTP Shift

假设完整序列被两个 CP Rank 切分：

```text
全局位置： 0 1 2 3 | 4 5 6 7
CP Rank 0：0 1 2 3
CP Rank 1：4 5 6 7
```

若某个 MTP Depth 要预测 `t+2`：

```text
位置 2 的 Label 是位置 4 → 位于 Rank 1
位置 3 的 Label 是位置 5 → 位于 Rank 1
```

若 Rank 0 只在本地 `[0,1,2,3]` 上做 `roll(-2)`，就可能把位置 2/3 错接回本地开头，形成没有报错但监督错误的 Silent Corruption。

正确思路有两类：

1. 在全局序列语义仍完整时构造各 Depth 的 Label/Mask/未来输入，再按 CP 布局切分；
2. 每个 Rank 与相邻 Rank 交换最大 Prediction Depth 所需的边界 Token/Embedding，形成 Halo，再做 CP-aware Shift。

具体框架可以选择不同实现，但验收结果必须与未切分的全局语义一致。

## 4. Packed Sequence 让边界多一层

一条物理序列里可能打包多个逻辑样本：

```text
[A B C | X Y Z]
```

对 `C` 来说，未来位置不能是 `X`；分隔线是样本边界，不是普通相邻 Token。MTP Mask 至少同时考虑：

- Padding；
- Prompt/Response 是否参与辅助 Loss；
- Sequence 尾部越界；
- Packed Segment 尾部；
- 截断；
- CP Rank 边界。

对每个 MTP Depth，Label、未来 Token Embedding 和 Mask 必须一起 Shift。只移动 Label、不移动 Mask，会让本应屏蔽的位置参与 Loss；只移动 Token、不传播 `padding_mask`，还可能让 MoE Router/GDN 把 Padding 当成真实 Token。

## 5. 为什么 Padding Mask 要贯穿整个调用链

一个可靠的调用链应类似：

```text
Batch / Packed Metadata
→ GPT Model
→ MTP Module
→ 每个逻辑 Depth 的 Shift
→ MTP Decoder Layer
→ Attention / GDN / MoE
```

Mask 不是只在最外层 Loss 时使用。它还可能影响：

- Attention 或 GDN 是否读取 Padding；
- MoE Router 是否给 Padding 分配 Expert；
- Token Dispatch 数量；
- Recompute 重跑时的输入一致性；
- 每层有效 Token 统计。

因此“模型 Forward 能接收 `padding_mask`”不等于 MTP 路径已支持；必须证明 Mask 传到每个 MTP Depth 和内部模块。

## 6. DP×CP 的全局 Token 归一化

每个 Rank 的有效 MTP Token 数可能不同。正确全局平均应是：

$$
L_{\text{mtp}}
=
\frac{
\sum_r\sum_{i\in\mathcal V_r}\ell_i
}{
\sum_r|\mathcal V_r|
}
$$

其中 $\mathcal V_r$ 是 Rank $r$ 的有效 Token 集合。

工程上更清楚的写法是：

```text
每个 Rank：local_numerator = 有效 Token Loss 之和
每个 Rank：local_count     = 有效 Token 数

跨 DP×CP：global_numerator = SUM(local_numerator)
跨 DP×CP：global_count     = SUM(local_count)

global_loss = global_numerator / global_count
```

不能简单平均每个 Rank 的 Local Mean，因为各 Rank 的有效 Token 数可能不同。

某个 CP Rank 因为恰好只拿到 Segment 尾部而 `local_count=0`，并不代表训练失败。只要全局 Count 大于零，且该 Rank 仍参与同样的 Collective，最终结果可以保持有限且一致。

## 7. Activation Recompute 的隐藏正确性要求

Recompute 会在 Backward 时重新执行部分 Forward 以节省 Activation 显存。MTP+CP 下，重跑必须复用同一批语义输入：

- 同一逻辑 Depth；
- 同一 Future Token/Embedding；
- 同一 Packed Boundary；
- 同一 Shift 后的 Padding/Loss Mask；
- 同一 CP 局部分片与通信结果。

如果正常 Forward 传了 Mask，但 checkpoint/recompute wrapper 丢了某个参数，第一次 Forward 可能正确，Backward 重算却走了不同路径，造成梯度错误或返回值解包错误。

所以验证不能只看 Forward 输出，还要让 full recompute 真正进入 Backward，并比较 Gradient 和一次 Optimizer Update。

## 8. Detach 与 CP 是两个正交维度

`detach_encoder` 决定 MTP Loss 是否回传到 Backbone：

```text
detach=true：
MTP Loss → MTP 专属参数

detach=false：
MTP Loss → MTP 参数 + Backbone/共享 Embedding/LM Head
```

CP 决定序列怎样分片及跨 Rank 通信。二者互不替代：

- Detach 正确不能证明 CP Label/Mask 正确；
- CP 数值一致也不能替代梯度边界检查；
- 若共享 Embedding/LM Head，仍需明确它们是否进入 Optimizer 和怎样归约梯度。

## 9. 为什么 CP8 仍可能在词表输出处 OOM

CP 将序列长度 $S$ 切成近似 $S/CP$，但不会自动切词表大小 $V$。若 Tensor Parallel 为 1，某些输出或梯度仍可能是：

$$
\left[\frac{S}{CP},V\right]
$$

其内存量近似：

$$
\text{bytes}
\approx
\frac{S}{CP}\times V\times\text{bytes-per-element}
$$

当 $S$ 和 $V$ 都很大时，即使 CP 已切序列，这个二维张量仍可能超过单卡余量。这里揭示了一个重要原则：

```text
CP 解决序列维度
TP 解决大矩阵/词表维度
PP 解决层数维度
EP 解决 Expert 维度
```

不同并行策略切不同轴，不能期待一个并行维度自动解决所有 OOM。

同时要分清：

- 32K 两步训练能证明 MTP+CP 正确性；
- 128K OOM 说明当前拓扑的容量边界；
- 它不自动推翻已经通过的小规模数值一致性，也不证明 128K 语义错误。

## 10. 线性 MTP 与树状能力

当前线性 MTP 的候选关系是：

```text
Draft 1 → Draft 2 → Draft 3
```

后一个候选依赖前一个候选，因此首个 Reject 后，后续候选的条件上下文也失效。

树状 Draft 则同时维护多个分支，需要额外处理：

- 分支拓扑；
- Tree Attention Mask；
- 多候选 Verify；
- 分支 Cache/State 提交；
- Top-K 分叉及训练标签。

所以“不支持 Tree Training”不是少一个开关，而是当前实现只承诺单链语义。线性 Prediction Depth K 不应被解释成每层有 K 个树分支。

## 11. 最小正确性验证阶梯

### 11.1 小序列显式单测

手写 8～16 个 Token 和 Segment Boundary，逐位置检查 K2/K3 Label、Future Input 和 Mask。

### 11.2 CP1 与 CP2 数值一致性

固定同一全局 Batch 和 Seed，比较：

- 拼回全局后的 Label/Mask；
- 每个 MTP Depth 的 Loss；
- 关键参数 Gradient；
- 一次 Optimizer Update。

### 11.3 边界测试

必须覆盖：

- Packed Segment 跨 CP 边界；
- 某个 Rank 本地零有效 MTP Token；
- K2/K3 不同 Halo 宽度；
- Full Recompute；
- Detach true/false 的 Gradient 边界。

### 11.4 完整模型 Smoke

再验证真实模型构建、Checkpoint 加载、两步训练、有限 Loss/Grad Norm 和参数更新。这里应把“功能正确性”和“目标长度容量”分成两个验收门。

当前证据边界：完整模型的 Packed CP8/K2、32K 两步训练已能观察到有限 Loss/Gradient 和参数变化，这支持“训练正确性闭环已跑通”。它不证明 128K 容量或性能已通过；更长序列仍可能在未分片的词表 Logits/Gradient 轴 OOM。

### 11.5 2026-08-28 补充：普通 CP 能力不等于 MTP+CP 集成能力

一个运行镜像或 Megatron-Core 版本“支持 CP”，只证明普通主干模型可能具备序列分片能力。MTP+CP 还依赖一组组合能力，例如：

- CP-aware Future Token/Embedding Shift；
- 将 Tensor 按 CP 布局移动或滚动的底层 API；
- MTP Loss 的 CP Group 与全局 Token 归一化；
- Packed Sequence Metadata 在 MTP/Recompute 中透传；
- 当前 MTP 模块所期望的返回值和 Loss 处理接口。

因此运行前应做 Capability Probe，而不是只比较版本字符串：

```text
普通 CP 可用？
→ MTP+CP 需要的函数、参数和 Process Group 是否都存在？
→ 调用签名与当前 Adapter 是否匹配？
├── 是：允许 CP>1 MTP Preflight
└── 否：Fail Fast，并要求使用已适配的运行镜像/底层实现
```

`CP=1` 不触发跨 Rank Future 依赖，所以它可以在缺少上述组合 API 时仍然工作。但：

```text
CP=1 MTP PASS
≠ 当前运行环境支持 CP>1 MTP
```

这类失败是 Runtime Capability/Integration 问题，不应误写成“算法上 MTP 不支持 CP”。

## 12. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| CP>1 只是性能优化 | Future Label/Embedding 会跨 Rank，是训练正确性问题 |
| `prediction_depth=2` 就有两个物理 MTP 层 | 可能只有一个物理层，被递归复用两次 |
| 只要 Label 对了就够了 | Future Input、Loss Mask、Padding、Packed Boundary 必须同步 |
| 平均每卡 Loss 就是全局 Loss | 必须按 DP×CP 全局有效 Token 数加权 |
| 某 Rank 零有效 Token 一定异常 | 可能是合法边界；关键是 Collective 一致和全局 Count |
| CP8 后长序列一定不会 OOM | CP 只切序列；词表/参数轴可能仍未分片 |
| 线性 K2 与两分支 Tree Draft 等价 | 一个是串行单链，一个是多分支候选结构 |
| 普通 CP 能跑就代表 MTP+CP 能跑 | 组合路径还需要 CP-aware Shift、Loss、Group 与 Packed/Recompute API |

## 13. 一分钟复习

1. MTP+CP 的本质是 Future 依赖跨 Rank，必须让 Label、Embedding、Mask 和 Packed Boundary 保持同一全局语义。
2. 全局 Loss 用 DP×CP 的 `sum(loss) / count(valid_tokens)`，不能平均各 Rank Mean。
3. Logical K 与 Physical MTP Layer Count 分离；一个物理层可重复形成多个逻辑 Depth。
4. Recompute 必须重放同一 Mask/Shift/分片语义，正确性要验证到 Backward 和 Optimizer。
5. CP 切 $S$，不切 $V$；正确性通过与目标长度容量通过是两个门。
6. 运行环境要按实际 API 做 Capability Probe；`CP=1` PASS 不证明 `CP>1` MTP 集成可用。

## 14. 自测问题

### 问题 1

为什么 CP Rank 0 不能直接在本地 Token 上用 `roll(-2)` 构造 MTP Label？

期望回答：边界位置的未来 Token 位于相邻 Rank，本地循环 Roll 会错接到本 Rank 开头，形成静默错误监督。

### 问题 2

一个 Rank 的有效 Token 数为零时，怎样避免 NaN 或 Collective 不一致？

期望回答：该 Rank 仍提供零 numerator/count 并参与同样的归约，最终使用全局 numerator/global count；全局 Count 必须大于零。

### 问题 3

为什么 CP8 下 `[S/CP,V]` 仍可能 OOM？

期望回答：CP 只缩小序列轴；若 TP=1，完整词表轴仍在每卡，长序列与大词表相乘后仍可能很大。

## 15. 与已有知识的联系

- [MTP Head 状态机与训练适配边界](../07/MTPHead状态机与训练适配边界_20260728.md)：旧文提出 CP>1 的跨 Rank Future 依赖，本文补齐正确实现和验证契约。
- [RL Trajectory 到 Megatron 与 MTP Loss 归一化](../07/RL轨迹到Megatron与MTP损失归一化_20260728.md)：旧文解释 Packed/Mask/Numerator-Count，本文扩展到 DP×CP 和 Recompute。
- [MTP 推测解码与成块验证](../07/MTP推测解码与成块验证_20260726.md)：补充线性 Draft Chain 与树状候选的能力边界。

## 16. 尚未解决与后续路线

- 深入 Megatron 中 CP 通信、Packed THD 布局和 GDN State 分片的具体实现。
- 对比全局预构造与 Halo Exchange 两种 Future Shift 方案的通信和内存成本。
- 学习词表并行 Cross-Entropy 如何避免完整 `[S,V]` Logits/Gradient 常驻单卡。

## 17. 公开参考材料

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-LM PR #2642](https://github.com/NVIDIA/Megatron-LM/pull/2642)
- [Megatron-LM PR #2645](https://github.com/NVIDIA/Megatron-LM/pull/2645)
