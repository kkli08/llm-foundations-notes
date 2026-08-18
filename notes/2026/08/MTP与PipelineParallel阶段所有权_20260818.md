# MTP 与 Pipeline Parallel 阶段所有权

> 日期：2026-08-18
> 来源：Codex Session `01a01362-656f-7630-9db5-8491f4bc4d9b`
> 状态：已整理

## 0. 这篇笔记最终要解决的问题

已有 MTP+CP 训练能力后，要支持 `PP>1`，是否只需删除一条限制？为什么 MTP 放在哪个 Pipeline Stage、谁计算 Loss、谁返回 Metrics、谁保存和同步参数，都会变成正确性问题？

## 1. 先给结论

1. PP 沿模型层切分，MTP 通常由最后一个 Pipeline Stage 持有，因为它依赖主干末端 Hidden State，并与 Final Norm/LM Head/Loss 相邻。
2. 支持 PP 不是删除 Guard：Pipeline Layout、Stage-local 构建/校验、Supervision、Metrics、Checkpoint 和 Rollout Weight Sync 都要遵守同一个 MTP Owner 契约。
3. Qwen3.5 K2 可能是两个逻辑 MTP Depth、一个物理参数层；Layout 的计算条目与实际参数数量不能混为一谈。
4. MTP Token Loss 的统计副本属于 DP×CP，不属于 PP；把 PP 加入 Loss 分母会算错，甚至导致 Collective 死锁。
5. 实现应面向通用 PP-N，但第一轮 GPU 正确性优先验证 PP2，再补 PP4；不能把 `pp_size==2` 写死，也不能把 PP2 通过当成任意 PP-N 已验证。
6. VPP 是 PP 内的交错虚拟 Stage。第一阶段保持 `VPP=1`，不要同时打开两种新的 Stage 所有权复杂度。

## 2. PP 和 VPP 分别是什么

### 2.1 Pipeline Parallel

PP 把 Transformer 层切成多个连续模型段：

```text
PP0：Embedding + 前半 Backbone
             │ Hidden State P2P
PP1：后半 Backbone + Final Norm + LM Head + Loss
```

每个 PP Rank 只持有自己的参数和 Activation。Microbatch 在 Stage 之间流水传递，以减少整模型放在单卡的显存压力。

### 2.2 Virtual Pipeline Parallel

VPP（Virtual Pipeline Parallelism）把每个物理 PP Rank 再拆成多个不连续的模型 Chunk，交错调度：

```text
PP=2, VPP=1：
Rank 0 持有连续前半段，Rank 1 持有连续后半段

PP=2, VPP=2：
两个物理 Rank 各持有两个模型 Chunk，形成四个 Virtual Stage
```

VPP 不增加 GPU 数量，主要用于减小 Pipeline Bubble，但会增加：

- Local Model Chunk 数；
- Forward/Backward 调度顺序；
- Stage Ownership；
- Checkpoint 遍历；
- Metrics 和 Weight Conversion 的映射复杂度。

因此第一阶段合理边界是：

```text
PP >= 2
VPP = 1
```

## 3. 为什么 MTP 通常属于最后一个 PP Stage

Native MTP 的输入通常包括主干末端 Hidden State、未来 Token Embedding，并继续经过 MTP Decoder Layer、Final Norm 和共享 LM Head。

```text
PP0：Embedding + Backbone 前半
              │
              ▼
PP1：Backbone 后半
     → MTP Depth 1 / Depth 2
     → Final Norm / LM Head
     ├─ Main RL Loss
     └─ MTP Loss
```

若把 MTP 放在前面 Stage，就需要把尚未完成的主干表示或共享输出层跨 Stage 来回传递，既违背默认模型结构，也增加额外 P2P 和所有权歧义。

所以一个清晰契约是：

- 非最后 Stage：不持有 MTP 参数，不计算 MTP Loss；
- 最后 Stage：精确持有预期的物理 MTP 参数，并负责 MTP Forward/Loss；
- Stage-local 校验必须知道自己是不是 Owner，不能要求每个 Rank 都找到 MTP Layer。

## 4. Pipeline Layout 不能漏掉 MTP

Pipeline Layout 决定每个 Stage 物化哪些层。普通 Layout 可能只描述：

```text
embedding | decoder layers | loss
```

启用 MTP 后还必须表达：

```text
embedding | decoder layers | mtp entries | loss
```

只删除 PP 门禁但不修改 Layout，可能出现：

```text
配置声称启用 MTP
→ 自定义 Layout 没有 MTP 条目
→ 所有 Stage 都没有物化 MTP
→ 后续加载、Loss 或同步才失败
```

这属于模型结构契约缺失，而不是普通运行参数错误。

## 5. 逻辑 MTP Depth 与物理参数层再次分离

对 repeated-layer K2：

```text
逻辑计算：MTP Depth 1 + MTP Depth 2
物理参数：只有一份 MTP Layer W
```

Pipeline Layout 和 Stage Balancing 关心的是计算次数，Checkpoint/Optimizer/显存关心的是物理参数数量。

因此可能需要同时记录：

| 维度 | 含义 |
|---|---|
| Logical Depth K | 每个 Step 执行多少次 MTP Forward、产生多少个 Loss |
| Physical MTP Layers | 实际有多少份不同参数、Checkpoint 保存多少层 |

若 Layout 只按一层物理参数估计，会低估最后 Stage 的计算负载；若加载器按 K 个逻辑条目要求 K 份参数，又会错误拒绝合法 Checkpoint。

## 6. Stage-aware 构建与 Fail-closed 校验

正确校验不是“所有 Rank 都必须有 MTP”，而是：

```text
if 当前 Rank 不是 MTP Owner：
    本地 MTP 层数必须为 0
    Configure MTP 是 no-op

if 当前 Rank 是 MTP Owner：
    本地物理 MTP 层数必须精确匹配 Checkpoint
    缺层、多层、Shape 不符都 Fail Closed
```

这样既允许模型分片，又不会用宽松 `strict=False` 掩盖最后 Stage 真正缺少 MTP 参数。

## 7. Supervision 和 Loss 的所有权

MTP Supervision 包括：

- 各逻辑 Depth 的 Future Label；
- Future Token Embedding；
- Loss/Padding/Packed Boundary Mask；
- DP×CP 的有效 Token Count。

它们只需在 MTP Owner Stage 进入 MTP Forward/Loss。每个 PP Stage 都重复构造会浪费内存，也会模糊 Collective 到底由谁调用。

最关键的是 Loss Group：

```text
DP：数据副本，需要归约
CP：同一序列的 Token 分片，需要合并 numerator/count
PP：同一个模型的不同层段，不是数据副本
```

因此：

$$
L_{\text{mtp}}
=
\frac{\operatorname{SUM}_{DP\times CP}(\text{valid loss sum})}
{\operatorname{SUM}_{DP\times CP}(\text{valid token count})}
$$

不能把 PP Group 加入分母。非最后 Stage 没有 MTP Loss，如果它们进入另一套不匹配 Collective，还可能产生 Hang。

## 8. 为什么 Metrics 必须从最后 Stage 回传 PP0

训练系统的 Controller 往往只保留某个 DP Head 的返回值，而该 Head 常位于 `TP0/CP0/PP0`。但 MTP Tracker 只在最后 PP Stage 有值。

如果不显式回传：

```text
MTP Forward/Loss/Backward 实际成功
→ 最后 Stage 有 mtp_1_loss / mtp_2_loss
→ PP0 返回给 Controller 的字典没有这些字段
→ 监控看起来像“没有训练 MTP”
```

稳定方案是：

1. 最后 Stage 按固定顺序收集 K 个标量；
2. 打包成固定 Shape 的 GPU Tensor；
3. 沿 PP Group Broadcast 到 PP0/所有 Stage；
4. PP0 再恢复成固定 Schema 的 Metrics。

不建议用动态 Object Collective，因为不同 Rank 的 Key 集合不同，容易产生 Collective 顺序不一致。

## 9. Checkpoint 与 Rollout Weight Sync

### 9.1 Checkpoint

DCP 可以保存 PP Shard，但还需验证：

- 最后 Stage 的 MTP 参数确实进入 Checkpoint；
- repeated-layer 只保存物理层一次；
- Shared Embedding/LM Head 的所有权正确；
- Save → Reload 后 MTP 输出和参数集合一致。

### 9.2 Online Weight Sync

PP0 和 PP1 分别导出自己持有的参数：

```text
PP0 → 前半 Backbone
PP1 → 后半 Backbone + MTP
```

Rollout 端最终必须得到完整 Exact Set：

```text
expected = converted = received = applied
```

需要特别验证多 PP Stage 的同步 Session 顺序、Layer Offset、Version Commit，以及只有最后 Stage 导出的 MTP Key 是否被正确合并。

## 10. Pipeline Balance 与 Bubble

平均分 Backbone Layer 不一定得到平均计算量，因为最后 Stage 还承担：

- Final Norm / LM Head；
- Main Loss；
- K 次 MTP Forward；
- MTP Loss 和 Metrics。

正式 Layout 可能需要给最后 Stage 少分几层 Backbone，按“实际计算时间”而不是“物理参数层数”平衡。

VPP 可以进一步减小 Bubble，但应在普通 PP-N 的 Ownership/Checkpoint/Metrics 稳定后单独引入。

## 11. 为什么实现通用 PP-N，却先验证 PP2

代码不应写死：

```text
pp_size == 2
last_rank == 1
```

而应使用通用语义：

- 当前 Rank 是否最后 Stage；
- 当前 Pipeline Group；
- 动态 `pipeline_parallel_size`；
- 根据 Layout 判断 MTP Owner。

PP2 能覆盖最重要的两种角色：非 Owner 与 Owner。但 PP4/PP8 还会暴露：

- 更多 Stage 的 Layout 切分；
- 空 Stage 或层数不可分；
- Source Rank 写死；
- Weight Sync Session 顺序；
- 更明显的 Stage Imbalance。

所以推荐：

```text
实现：通用 PP-N
单测：至少 PP2 + PP4 Layout
首轮 GPU：PP2 / VPP1
后续 Correctness：补 PP4
```

## 12. PP 与 CP 的复杂度为什么不同

| 维度 | CP 的主要难点 | PP 的主要难点 |
|---|---|---|
| 切分对象 | Token/序列 | 模型 Layer |
| 核心风险 | Future Shift、Packed Mask、Token Count | Stage Ownership、Layout、Metrics、Checkpoint/Sync |
| 错误类型 | 数值监督静默错误 | 模块缺失、结果不可见、状态不完整 |
| 主要复杂度 | 算法与张量语义 | 系统集成与状态所有权 |

PP 不一定在数学上比 CP 更难，但牵涉模块更分散，因此代码改动面看起来更大。

## 13. 最小验证阶梯

### 13.1 Layout/Owner 单测

- PP2/K1：只有最后 Stage 有 MTP；
- Repeated K2：两个逻辑条目、一份物理参数；
- PP0 零层合法；
- Owner 缺层必须失败；
- PP4 Layout 无空 Stage 和硬编码 Rank。

### 13.2 单节点 PP2 Correctness

- PP2/TP/CP 的合法组合；
- Packed THD、K2、至少两步训练；
- Main/MTP Loss、Gradient、Parameter Delta；
- PP0 能看到 MTP Metrics；
- Full Recompute。

### 13.3 状态闭环

- DCP Save/Reload；
- Rollout Version 0→1；
- Target/Drafter 参数 Exact Set；
- 下一轮 Rollout 使用新版本。

### 13.4 扩展验证

- PP4 Correctness；
- PP1/PP2 Matched 性能；
- 最后再研究 VPP>1、VLM 和更复杂模型家族。

## 14. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| 删除 `PP>1` Guard 就支持 MTP+PP | Layout、Owner、Metrics、Checkpoint/Sync 都必须补齐 |
| 每个 PP Rank 都应有 MTP 层 | 通常只有最后 Stage 持有物理 MTP 参数 |
| PP 也应加入 MTP Token Loss 分母 | PP 切模型，不复制数据；Loss 归一化仍是 DP×CP |
| K2 表示最后 Stage 有两份参数 | Repeated K2 可只有一份物理参数、两次逻辑调用 |
| PP2 跑通就证明 PP8 | PP2 覆盖核心角色，但不覆盖多 Stage Layout/同步顺序 |
| VPP=2 就是多用两倍 GPU | VPP 不增加物理 Rank，而是增加交错 Model Chunk |

## 15. 一分钟复习

1. MTP 通常属于最后 PP Stage；非 Owner 必须零层，Owner 必须精确持有物理参数。
2. Pipeline Layout 要表达逻辑 MTP 计算；Checkpoint 要表达物理参数，两者不能共用一个“层数”概念。
3. MTP Supervision/Loss 在 Owner Stage，Token 归一化只跨 DP×CP，不跨 PP。
4. 最后 Stage 的 MTP Metrics 必须显式传回 PP0/Controller。
5. 实现面向 PP-N，首轮验证 PP2/VPP1，再补 PP4；VPP 独立分阶段。

## 16. 自测问题

### 问题 1

为什么最后 PP Stage 有 MTP Loss，但训练面板可能看不到？

期望回答：Controller 可能只读取 PP0/DP Head 返回，而 Tracker 位于最后 Stage；必须用固定 Schema 把 Metrics 沿 PP Group 回传。

### 问题 2

为什么 MTP Loss 的分母不能再乘 PP Size？

期望回答：PP Rank 是同一个模型的不同层段，不是独立 Token/Data 副本；只有 DP×CP 构成全局有效 Token 集合。

### 问题 3

Repeated K2 对 Pipeline Balance 和 Checkpoint 的含义分别是什么？

期望回答：Balance 要考虑两次逻辑 Forward 的计算，Checkpoint 只保存一份物理 MTP 参数。

## 17. 与已有知识的联系

- [MTP 与 Context Parallel 训练正确性](MTP与ContextParallel训练正确性_20260817.md)：CP 解决 Token 分片和监督语义；本文补充模型 Stage 所有权。
- [MTP 模型状态流与在线权重事务](../07/MTP模型状态流与在线权重事务_20260729.md)：本文把 Exact Set 和 Version Transaction 扩展到多个 PP Stage。
- [训练与 RL 后训练基础](../07/训练与RL后训练基础_20260724.md)：回看 DP/TP/PP/EP/CP 各自切分的维度。

## 18. 尚未解决与后续路线

- VPP>1 下多个 Local Model Chunk 的 MTP Ownership 和 Metrics。
- 最后 Stage MTP 负载的自动 Pipeline Layout 优化。
- VLM、多模态 Embedding 和 Qwen3-MoE Checkpoint-native MTP 的 PP 适配。
