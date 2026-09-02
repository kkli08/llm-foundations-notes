# MTP 精度 Parity 评测与统计边界

> 日期：2026-09-02
>
> 来源：近期同 Checkpoint、不同 Speculative K 的质量评测复盘
>
> 状态：已整理

## 0. 先给结论

MTP/Speculative Decoding 的首要质量门禁通常不是“精度提高”，而是：

> 在只改变推理侧 Speculative 配置时，最终 Target 分布与任务质量没有出现不可接受的系统性退化。

如果 MTP-off、K1、K2、K3 的分数接近且不随 K 单调变化，可以形成“工程上质量基本持平”的方向性结论；但不能自动宣称：

- 数学上完全等价；
- 统计学等价；
- MTP 提高了精度；
- MTP 一定提高吞吐或降低延迟。

## 1. 严格对照必须固定什么

最干净的设计是同一 Checkpoint：

```text
同一 Model Weights
同一 Tokenizer
同一 Dataset / Prompt Template
同一 Judge / Scorer
同一 TP/并发
同一 Sampling 参数
同一最大输出长度

只改变：
MTP off / speculative K
```

如果换了 Checkpoint，就同时改变了训练步数、Drafter 参数和 Target 参数，无法把差异归因给推理 K。

## 2. 训练 Depth D 与推理 K 要分开记录

```text
Train Depth D：训练时监督多少个未来深度
Rollout K：推理时连续提出多少个 Draft Token
```

同一物理 MTP Layer 可以递归调用多次，所以可能出现：

```text
Train D=1
Rollout K=2 或 K=3
```

质量评测报告若只写“MTP=2”，无法判断它指训练深度、物理层数还是推理 Draft 数。

## 3. 为什么分数接近仍不能直接说“等价”

“没有看到明显差异”只代表当前实验没有发现差异。统计等价需要预先定义可接受边界：

```text
Δ = score_mtp - score_off
```

先规定工程容忍区间：

```text
[-ε, +ε]
```

再检查置信区间是否整体落入该区间。若只是点估计接近，但置信区间很宽，结论应是：

> 当前没有观察到稳定退化，但证据不足以证明统计等价。

这与传统“差异显著性检验不显著”也不同：

```text
未拒绝有差异
≠ 已证明等价
```

## 4. Temperature Sampling 会引入额外方差

当 `temperature > 0` 时，同一 Prompt 可能产生不同回答。单次运行的差异可能来自 Sampling，而不是 MTP K。

更稳妥的方案：

### 方案 A：固定或贪心解码

减少随机性，适合做严格功能 Parity。

### 方案 B：多 Seed 重复

```text
每个 Config × 多个 Seed
→ 报告 Mean / Std / Confidence Interval
```

### 方案 C：逐题配对

同一题目比较 off/on 是否同时答对：

| | MTP 对 | MTP 错 |
|---|---:|---:|
| Off 对 | 都对 | 只 Off 对 |
| Off 错 | 只 MTP 对 | 都错 |

整体 Accuracy 接近时，“只 Off 对”和“只 MTP 对”是否平衡，比只看两个均值更有信息量。

## 5. 小数据集为什么特别不稳定

若数据集只有 \(N\) 题，一道题对 Accuracy 的影响是：

\[
\frac{1}{N}\times 100\%
\]

例如很小的数据集里，一两题就能造成数个百分点波动。汇总多个数据集时，还要说明聚合方式：

### Macro Average

每个数据集权重相同：

\[
\frac{1}{D}\sum_d score_d
\]

### Micro Average

每道样本权重相同：

\[
\frac{\sum_d correct_d}{\sum_d N_d}
\]

两个指标可能给出不同印象。小数据集在 Macro Average 中的权重会被放大。

## 6. “不随 K 单调”说明什么

若结果表现为：

```text
off ≈ K1 ≈ K2 ≈ K3
且 K 增大时分数上下小幅波动
```

较合理的解释是：

- 没有观察到随 K 增大而持续恶化的系统性信号；
- 小差异更像 Sampling、样本量或 Judge 方差；
- 没有证据表明更大的 K 提高任务精度。

不能解释成：

```text
K 越大，模型越聪明
```

MTP 的目标是减少 Target 串行 Decode 次数，不是改变 Target 模型能力。

## 7. 为什么理论等价在工程上仍需验证

理想 Speculative Decoding 会由 Target Verify 保证最终采样分布正确。但工程实现仍可能在以下位置出错：

- Sampling Correction；
- Accept/Reject；
- 最长前缀提交；
- KV/GDN State Commit/Rollback；
- EOS/长度边界；
- Batch/Continuous Batching；
- Draft/Target Tokenizer；
- Random Seed 与 RNG 消费顺序。

所以质量 Parity 是实现正确性门禁，而不是多余测试。

## 8. 质量、性能和接受率是三张表

### 质量表

- Accuracy / Pass Rate；
- Per-dataset Score；
- Paired Win/Loss；
- 多 Seed Mean/Std；
- 置信区间。

### 性能表

- Output TPS；
- TPOT / TTFT；
- End-to-end Latency；
- Peak VRAM；
- 不同 Length / Local BS。

### MTP 机制表

- Acceptance Rate；
- Acceptance Length；
- Per-depth Acceptance；
- Draft/Verify/Sampling 时间。

质量持平不证明性能变快；接受率高也不保证端到端加速。

## 9. Checkpoint 能力要检查真实 Tensor

Checkpoint Index 可能残留旧的 MTP Key，但实际 Shard 没有对应 Tensor。开启 MTP 前应核验：

```text
Index 声明集合
实际 Shard Tensor 集合
Loader Expected 集合
Shape / dtype
```

只有 Index Key 不足以证明该 Checkpoint 可做 MTP-on。严格门禁仍是：

```text
expected = indexed = physically present = loaded
```

若只有 Target 权重，应明确把它当作 MTP-off Checkpoint，而不是用宽松 Loader 猜测。

## 10. 报告阶段失败不等于模型评测失败

一次评测通常包含：

```text
Model Generation
→ Judge / Scoring
→ Aggregation
→ Report Generation
```

若 Generation/Judge 已完成，只是 Report 阶段因配置或临时服务问题失败，重启报告并不会重新证明模型质量，也不应把它记成模型推理失败。

记录时要写清失败阶段：

- Execution Failure；
- Judge Failure；
- Aggregation Failure；
- Report Failure。

## 11. 推荐结论模板

### 证据较弱时

> 在当前单次、同 Checkpoint 对照中，没有观察到随 Speculative K 增大而单调变化的质量信号；各配置点估计接近。该结果支持继续进行性能评测，但不构成统计等价或精度提升证明。

### 证据较强时

> 在固定解码或多 Seed、逐题配对的评测中，MTP-on 与 off 的差值置信区间位于预先设定的等价区间内，因此在该任务集和配置下通过质量 parity 门禁。

## 12. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| 两个 Accuracy 接近就完全等价 | 还要看方差、置信区间和等价边界 |
| p-value 不显著就证明等价 | 未发现差异不等于证明等价 |
| MTP K 越大精度应越高 | K 是推理候选深度，目标是性能而非能力增强 |
| 同模型家族即可公平对比 | 最好同一 Checkpoint，只改 Speculative 配置 |
| 质量持平就证明加速 | 仍需独立的性能和接受率表 |
| Acceptance 高就证明质量正确 | 还需验证最终 Target 分布和任务得分 |
| Index 有 MTP Key 就能开启 MTP | 必须检查实际 Tensor 与 Loader Exact Set |
| Report 失败就是模型评测失败 | 要定位 Generation/Judge/Aggregation/Report 阶段 |

## 13. 一分钟复习

```text
MTP 质量 Parity：
同一 Checkpoint
→ 只改变 off / K
→ 固定 Dataset、Sampling、Judge、并发
→ 看逐题配对 + 多 Seed/置信区间
→ 预先定义等价边界 ε

质量表 ≠ 性能表 ≠ Acceptance 表
```

## 14. 自测问题

1. 为什么同一模型家族、不同 Checkpoint 不足以隔离 MTP K 的影响？
2. “差异不显著”和“统计等价”有什么区别？
3. Temperature Sampling 下为什么要多 Seed 或补固定解码？
4. Macro 与 Micro Average 为什么可能给出不同结论？
5. 质量持平、Acceptance 高和端到端加速为什么必须分开证明？
6. Checkpoint Index 有 MTP Key，为什么仍不能直接打开 MTP？

## 15. 与已有知识的联系

- [推理性能的配置—执行证据链](../08/推理性能的配置执行证据链_20260812.md)：请求配置、运行时事实与最终测量需要逐层核验。
- [Online RL 的 Cohort 并发与会话容量](../08/OnlineRL的Cohort并发与会话容量_20260818.md)：已有严格 MTP on/off A/B 的变量固定原则。
- [On-policy 训推与 MTP 性能分析](../08/OnPolicy训推与MTP性能分析_20260806.md)：质量、Acceptance 与吞吐是不同观测维度。
- [MTP 模型状态流与在线权重事务](../07/MTP模型状态流与在线权重事务_20260729.md)：Checkpoint 能力需要 Index、Tensor、Loader 与 Version 的 Exact-set 证明。
