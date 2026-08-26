# Online MTP 训练成本、K/D 解耦与性能归因

> 日期：2026-08-26
> 来源：近期 Online MTP、Profiler、Router Replay 与正式训练对话的增量整理
> 状态：已整理

## 整理记录

- 2026-08-26：整理近期活跃任务中可复用的理论与性能归因方法。
- 工作任务、内部地址、Job ID 和原始日志未写入本知识库。

## 0. 这篇笔记最终要解决的问题

1. Rollout 的 MTP Draft 步数和训练的 MTP Prediction Depth 为什么可以不同？
2. MTP-Off、Frozen MTP、Online MTP 三组实验分别能回答什么？
3. 为什么 Online MTP 可能缩短 Rollout，却让整个 RL Step 变慢？
4. Actor `recompute_logprob`、Activation Recompute 和真正的 PPO Forward/Backward 有什么区别？
5. Cross-Entropy Fusion 到底融合了什么，为什么 LM Head 反向仍可能成为大头？
6. Profiler 中很长的 CPU/GPU 区间、NCCL 区间和空白区间应该怎样解释？
7. Checkpoint 恢复、HF 权重 Warm Start 和完整断点续训为什么不是一回事？

## 1. 先给结论

1. **Rollout K 与 Train Depth D 是两个独立旋钮。** Rollout 可以递归调用同一份 Drafter 多次得到 K 个候选；训练不必为每个候选深度都构造一份辅助 Loss。
2. **Online MTP 本质上是随当前 Policy Rollout 同步进行的辅助 Token CE/SFT。** RL Loss 更新主干，MTP Loss 用当前 Rollout Token 作 Hard Label，通常只更新 MTP 分支。
3. **MTP-Off、Frozen、Online 必须三臂比较。** Off→Frozen 看纯投机推理收益；Frozen→Online 看在线适配的增益与训练成本；Off→Online 看端到端总价值。
4. **`detach_encoder=true` 只切断 MTP Loss 对主干共享部分的参数梯度，不会消除 MTP Forward、MTP Backward、LM Head 输入梯度和 Activation Recompute。**
5. **Native CE Fusion 通常只融合 Softmax/归约/Cross-Entropy 链，不自动融合全词表 LM Head Projection。** 冻结 LM Head 权重可以省掉 WGRAD，但为了把梯度传回 MTP Hidden State，DGRAD 仍要计算。
6. **Actor Logprob 重算与 Activation Recompute 不是一回事。** 前者是 RL 算法阶段的一次 Forward-only；后者是训练反向时用计算换激活显存。
7. **Profiler 是归因工具，不是正式吞吐结果。** CPU Envelope、GPU Kernel、Collective 等待、跨 Stream 重叠和调度空洞必须分开统计。
8. **MoE All-to-All 很长不等于网络真的搬了那么久。** 它可能主要在等待晚到 Rank；瞬时 Expert 不均衡、上游 Host/Allocator Stall 和 Rank Arrival Skew 都可能放大 Collective 尾延迟。
9. **HF 权重恢复通常只是 Weight Warm Start。** 若没有 Optimizer、LR Scheduler、RNG、Dataloader Position 和全局步状态，就不能称为精确续训。

## 2. Online MTP 到底在训练什么

把主干参数记为 \(\theta\)，MTP/Drafter 参数记为 \(\phi\)。一个常见目标可以写成：

\[
L
=
L_{\mathrm{RL}}(\theta)
+
\lambda_{\mathrm{mtp}}
L_{\mathrm{MTP}}
\left(
\phi;
\operatorname{stopgrad}(h_\theta,E,W_{\mathrm{lm}})
\right)
\]

直观理解：

```text
当前 Policy Rollout 出一条 Response
        │
        ├─ Reward / Advantage → RL Loss → 更新主干 Actor
        │
        └─ Response Token 本身 → Future-token CE → 更新 MTP Drafter
```

它被称为“Online SFT”，是因为：

- 监督标签来自当前 Policy 刚生成的 Token，而不是固定离线 SFT 数据集；
- MTP 仍然做 Token-level Cross-Entropy，形式上类似 SFT；
- Reward 不直接决定某个 MTP Token Label 是否正确；
- 目标是让 Drafter 追随不断变化的 Target Policy，而不是让 MTP 单独学习任务奖励。

因此，Online MTP 不是另一套 RL 算法，而是 RL Actor 训练旁边的一条辅助监督分支。

## 3. Rollout K 与 Train Depth D 为什么必须分开

### 3.1 两个符号各自表示什么

```text
Rollout K：推理时连续提出多少个 Draft Token
Train D：训练时构造多少个未来 Offset 的 MTP Loss
```

例如：

```text
Rollout K=3：
同一个 MTP 模块递归执行三次
→ Draft x1、x2、x3

Train D=1：
只训练第一条未来预测链
→ 一次 MTP Forward + 一份 CE
```

同一份物理 MTP Layer 可以在推理时被重复调用多次。**逻辑执行次数不等于物理参数份数，也不等于必须训练相同数量的 Loss。**

### 3.2 为什么不应强制 K=D

若强制 `Rollout K = Train D`，会把两个不同目标绑死：

- Rollout K 主要控制接受长度与 Target Forward 摊销；
- Train D 主要控制辅助监督强度、显存和 Actor 更新成本。

更合理的搜索空间是：

| 配置 | 目的 |
|---|---|
| K0/D0 | 完全关闭 MTP 的基线 |
| K2/D0 | Frozen Drafter，只观察投机推理 |
| K2/D1 | 保留 K2 Rollout，但只支付一层在线训练成本 |
| K2/D2 | 两个逻辑深度都训练，监督最完整但成本最高 |

一个很重要的工程结论是：**先用 K2/D1 验证接受长度能否随训练改善，再决定 D2 是否值得。**

## 4. Off、Frozen、Online 三臂实验怎样解释

| 实验臂 | Actor 训练 MTP | Rollout 使用 MTP | 能回答什么 |
|---|---:|---:|---|
| MTP-Off | 否 | 否 | 无投机解码的总基线 |
| Frozen MTP | 否 | 是 | 纯投机推理收益和固定 Drafter 的表现 |
| Online MTP | 是 | 是 | 在线适配后的收益，以及为此付出的训练成本 |

三种差分口径：

### 4.1 Off → Frozen

固定训练侧没有 MTP Loss，只在 Rollout 打开投机解码。

```text
回答：仅靠已有 Drafter，Rollout 能快多少？
```

### 4.2 Frozen → Online

两边 Rollout 都使用同样的 K，只改变 MTP 是否参与在线训练。

```text
回答：在线适配让接受长度/吞吐增加多少？
代价：Actor 多支付多少 Forward、Backward、Recompute 和 CE？
```

### 4.3 Off → Online

这是最终产品价值，但它同时混入投机推理和在线训练两类变化。

```text
回答：整个 RL Step 最终快还是慢？
不能单独用于解释收益来自哪里。
```

Frozen Drafter 在短窗口内没有明显退化，并不能证明 Online MTP 没价值。它也可能说明：

- 初始 Drafter 已经较强；
- Policy 更新被 Clip，短期 Drift 小；
- Drafter 读取了当前 Target Hidden State 或共享参数，仍能部分跟随；
- 汇总均值掩盖了特定 Prompt、长度或训练阶段的局部退化。

所以更谨慎的表述是“Online MTP 持续增强或维持 Drafter”，而不是没有证据就断言“修复了明显衰减”。

## 5. 两种 Recompute 必须严格区分

### 5.1 Actor `recompute_logprob`

Rollout 完成后，Actor 需要在训练框架中重新计算当前/近端策略对已生成 Token 的 Logprob：

```text
Trajectory Token
→ Actor Forward-only
→ Logits / Logprob
→ PPO Ratio / Loss 的输入
```

这一阶段通常不构造 MTP Label，也不需要辅助 MTP Loss。如果旧实现复用了“启用 MTP 的完整模型 Forward”，就可能额外执行：

- MTP Block；
- MTP Postprocess；
- 不会被 Logprob 使用的辅助路径。

安全优化边界是：

```text
Forward-only Logprob 重算：只执行 Backbone
真正 PPO Forward/Backward：仍执行并训练配置的 MTP Depth
```

这属于移除算法阶段里的冗余分支，不等于关闭 Online MTP 训练。

### 5.2 Activation Recompute

Activation Recompute 是显存优化：Forward 时不保留所有中间激活，Backward 前重新执行部分 Forward。

```text
第一次 Forward：得到 Loss，但少存激活
Backward：需要某层激活
→ 再 Forward 一次恢复激活
→ 计算梯度
```

若一个物理 MTP Layer被两个逻辑 Depth 共用：

```text
原始 MTP Forward：D1 一次 + D2 一次
Backward Recompute：D1 一次 + D2 一次
```

所以一个物理层每个 Microbatch 可能执行四次。这里的“重复”是训练计算/显存权衡，不是 Actor Logprob 重算。

## 6. 为什么 Qwen3.5 的 MTP 训练可能很贵

### 6.1 主干便宜，不代表 MTP 相对便宜

混合架构主干可能大部分使用 GDN/Linear Attention，周期性插入 Full Attention；而原生 MTP Layer 自身可能是：

```text
Full Attention + MoE + Full-vocabulary LM Head
```

因此增加一个 MTP Depth，不能只按“主干层数的 1/N”估算。它可能相对主干平均层更贵。

### 6.2 每个 Depth 的主要成本

```text
Future Embedding / Shift / Mask
→ MTP Full Attention
→ MTP MoE（含 EP All-to-All）
→ Final Norm
→ 全词表 LM Head Projection
→ Vocab-parallel Cross-Entropy
→ Backward
→ Activation Recompute（若开启）
```

长序列、超大词表和动态 Padding 会一起放大这些成本。

### 6.3 CP 切序列，不自动切词表

Context Parallel 主要切 \(S\) 轴：

```text
[B, S, H] → 每个 CP Rank 处理一段 S
```

LM Head 产生的是：

```text
[local_tokens, vocab_shard]
```

词表轴是否切分由 TP/Vocab Parallel 决定。因此 CP 能降低本 Rank Token 数，但大词表 Logits 仍可能成为显存和计算瓶颈。

### 6.4 Mask 晚应用仍可能支付算力

若实现先为所有 Padding Token 计算 Projection/CE，最后才用 Mask 把 Loss 置零，那么：

```text
数学结果正确
≠
Padding 没有计算成本
```

这解释了为什么动态长度 Batch 被固定补到大长度时，MTP 成本可能非线性放大。

## 7. CE Fusion 的真实边界

### 7.1 未融合路径

一个朴素 Vocab-parallel CE 可能包含：

```text
Logits
→ Max
→ Subtract
→ Exp
→ Sum
→ Log
→ Gather Correct-class Logit
→ Cross-Entropy
```

Native CE Fusion 可以把这些小算子、归约和中间 Tensor 访问合成较少的编译/Triton Kernel。

### 7.2 它通常没有融合什么

CE Fusion 不等于：

```text
Hidden State × LM Head Weight
→ Cross-Entropy
```

被整个融合成一个 Kernel。全词表 Projection GEMM 往往仍独立存在。

### 7.3 冻结 LM Head 为何仍有大 GEMM

Linear 层反向有两类梯度：

```text
WGRAD：对 LM Head 权重求梯度
DGRAD：对输入 Hidden State 求梯度
```

冻结 LM Head 或对其 Detach：

- 可以不计算/不保存 WGRAD；
- 但 MTP Loss 要训练前面的 MTP Block，仍需 DGRAD 把梯度传回 Hidden State。

因此，Profiler 中看到大的 Projection Backward GEMM，不代表冻结失败；先判断它是 WGRAD 还是 DGRAD。

## 8. Profiler 应怎样做因果归因

### 8.1 三类时间不能直接相加

| 记录 | 代表什么 | 常见误区 |
|---|---|---|
| CPU Range/Envelope | Python/ATen 调用到返回的墙钟窗口 | 把异步等待全部当成 CPU 计算 |
| GPU Kernel Duration | 某 Stream 上 Kernel 真正执行时间 | 忽略其他 Stream 重叠 |
| Collective Range | 通信 Kernel + 等待参与 Rank | 全部当成纯网络搬运 |

正确方法是看：

- All-stream GPU Busy Union；
- CPU→CUDA Correlation/Flow；
- 每个 Kernel 的 Stream；
- 关键阶段的前后依赖；
- 真正的 GPU Idle Gap。

### 8.2 `Torch-Compiled Region` 不等于正在编译

Profiler 中很长的 `Torch-Compiled Region` 或 `Call CompiledFxGraph` 常表示“正在执行已编译图”。只有看到 Dynamo/Inductor Compile Event，才能说时间花在编译上。

预热可以减少冷启动编译，但不能消除：

- 已编译图内部的 GEMM；
- NCCL 等待；
- Host/Allocator Stall；
- 数据依赖导致的串行执行。

### 8.3 Profiler 结果不等于生产吞吐

正式性能结论至少需要：

```text
Profiler Off
+ 同一有效负载
+ 稳态多步
+ Token 归一化
+ Mean / Median / Tail
```

Profiler On 的单 Microbatch Trace 适合回答“时间去哪了”，不适合直接声明“配置 A 比配置 B 快多少”。

## 9. MoE EP All-to-All：Payload 与等待必须分开

MoE 每层大致经历：

```text
Router Top-K
→ 按 Expert 统计 Split Size
→ Dispatch All-to-All
→ Local Expert Compute
→ Combine All-to-All
```

当某次 All-to-All Envelope 很长时，有三种不同问题：

1. **真实 Payload 大或链路慢**：所有 Rank 大致同时进入，Kernel 本身都很长；
2. **瞬时 Expert Imbalance**：部分 Rank 收到更多 Token，Expert Compute 更久；
3. **Rank Arrival Skew**：早到 Rank 在 Collective 内等待，晚到 Rank 可能被上游 Allocation、Host 调度或别的算子阻塞。

一个典型信号是：

```text
早到 Rank：Collective Envelope 数秒
晚到 Rank：真正传输 Kernel 只有几十毫秒
```

这时不能说“网络传输了数秒”；直接证据是 Rank 到达不齐。Expert 不均衡可能是上游原因之一，但还要查晚到 Rank 在进入 Collective 前做了什么。

平均负载平衡也不能排除 Tail：

```text
长期每个 EP Rank 的平均 Token 数接近
但个别 Wave 的 max/min 很大
→ P95/P99 Layer Time 仍被拖慢
```

因此要同时报告平均、P50、P95 和最坏 Wave。

## 10. CUDA Memory History / Snapshot 能证明什么

PyTorch CUDA Memory History 记录 Caching Allocator 的分配、释放和栈信息，适合定位：

- 哪段代码申请大块显存；
- Reserved/Allocated 的变化；
- Fragmentation 或频繁申请的线索；
- 某个训练阶段是否产生异常峰值。

但它有明确边界：

- 看不到所有 NCCL 或直接 CUDA 分配；
- Snapshot 只有在 Dump 执行完成并落盘后才是有效证据；
- 节点失联导致训练中断、没有 Snapshot，不等于已经证明 CUDA OOM；
- 开启大量历史记录本身会增加 CPU 内存、Python 栈记录和 Host 开销，需要小心控制窗口和条目数。

因果判断应写成：

```text
已证明：记录已启动，随后节点失联，Snapshot 未完成
未证明：OOM、Fragmentation、Memory History 一定导致节点失联
```

不要用“没有产出 pickle”倒推出“Dump API 出错”；它也可能只是故障发生在 Dump 之前。

## 11. Acceptance Length 与 Acceptance Rate

设每个 Verify Step 提出 \(K\) 个 Draft Token，累计提出 \(P\) 个，接受 Draft Token 数为 \(A\)。

常见口径：

\[
\text{accept\_rate}=\frac{A}{P}
\]

有些框架报告：

\[
\text{accept\_length}
=1+\frac{A}{\text{verify steps}}
\]

这里的 1 是每次 Target Verify 至少正式推进的 Bonus/Target Token。因此：

```text
accept_length = 1.82
```

不表示 82% 接受率，而表示平均每次 Verify 正式推进 1.82 个 Token。

要解释在线训练是否让 Drafter 变好，至少需要：

- Accept Rate；
- Accept Length；
- 每个 Draft Depth 的分位置接受率；
- Policy Version / Training Step；
- 固定或分桶的 Prompt/Response Length。

MTP Loss 下降不能替代这些推理指标。

## 12. Agentic 场景的 MTP Mask

假设 \(g_i=1\) 表示 Token \(i\) 属于可监督的模型生成区间。第一层 MTP 使用中间 Token \(t+1\) 去预测目标 \(t+2\)：

\[
m_t=g_{t+1}\land g_{t+2}
\]

更深链路需要对沿途所有 Token 的合法性累计求交。

只把最终 Target Mask 向前 Shift，可能跨越：

- Assistant → Tool；
- Tool → Assistant；
- Packed Sample Boundary；
- 部分 Loss Mask 为 0 的区间。

所以 Agentic-safe MTP Mask 不是“只看最终 Label 是否有效”，而是“整条 speculative chain 是否都处于同一合法生成段”。

## 13. Checkpoint：精确续训与 Weight Warm Start

### 13.1 完整断点续训需要什么

```text
模型权重
+ Optimizer State
+ LR Scheduler State
+ RNG State
+ Dataloader Position
+ Global Step / Policy Version
+ 训练框架必要状态
```

### 13.2 只有 HF 权重时发生什么

```text
加载最新模型参数
→ Optimizer 重新初始化
→ RNG / 数据顺序从头开始
→ 只把日志步数偏移到旧位置
```

这叫 Weight Warm Start，不是精确 Resume。一个很容易误判的现象是：逻辑 Step 看似连续，但 Token 数、序列长度和耗时突然回到最初几步的分布。这通常是数据位置重启，不是性能退化。

Checkpoint Index 里若保留了已关闭分支的陈旧 Key，还要区分：

- Index 声明了什么；
- 实际 Shard 里有什么；
- Loader 是否允许该模式下缺少被禁用分支；
- 是否存在任何额外或错误 Shape 的 Tensor。

不能用全局 `strict=False` 把真正的参数缺失一起掩盖。

## 14. 推荐的性能验证阶梯

### 第一级：功能正确性

- MTP Depth、Loss Key 和参数更新符合配置；
- 只有合法 Token 进入 MTP Loss；
- 权重能同步到 Rollout Drafter；
- 2-Step 闭环证明新版本被下一轮 Rollout 使用。

### 第二级：端到端三臂对照

- Off / Frozen / Online；
- 固定 Rollout K、Batch、并行拓扑、长度和 Sampling；
- 同时记录 Rollout、Actor、Weight Sync、Total Step。

### 第三级：K/D 消融

- K2/D0、K2/D1、K2/D2；
- 观察接受率、接受长度、Actor 成本和总 Step。

### 第四级：固定输入 Profiler

- 先预热；
- 抓一个稳态 Microbatch；
- 用 NVTX/阶段标记拆 MTP Attention、MoE、Projection、CE、Backward/Recompute；
- 用同一份 Actor-ready Tensor 才能做严格 Kernel A/B。

### 第五级：针对性诊断

- Allocator 问题：CUDA Memory History/Snapshot；
- Kernel 带宽/算力：NCU；
- Host 调度：CPU Sampling；
- Rank Arrival/Collective：跨 Rank Correlation 与 Split Size。

## 15. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| Rollout K2 就必须训练 D2 | 推理递归深度和训练监督深度可以独立 |
| Online MTP 是另一种 RL | 它是 RL 旁边的在线辅助 Token CE/SFT |
| `detach_encoder` 后 MTP 很便宜 | 它只改变梯度边界，不消除 Forward、DGRAD 和 Recompute |
| CE Fusion 会消除 LM Head | 一般只融合 CE 链；全词表 Projection 仍在 |
| 冻结 LM Head 后不应有 Projection Backward | WGRAD 可省，传给 MTP Hidden State 的 DGRAD 仍需要 |
| `Torch-Compiled Region` 很长就是编译慢 | 它也可能只是执行已编译图 |
| NCCL Range 很长就是网络慢 | 可能主要在等待晚到 Rank |
| 平均 Expert 负载平衡就没有尾延迟 | 瞬时 Wave 仍可能严重不均衡或到达偏斜 |
| 没有 Memory Snapshot 就证明 Snapshot API 失败 | 也可能是任务在 Dump 前因别的原因中断 |
| 只恢复 HF 权重就是无缝续训 | 缺少优化器、RNG、数据位置时只是 Weight Warm Start |
| MTP Loss 降低就证明 Rollout 更快 | 还要看 Accept Length/Rate 和实际阶段耗时 |

## 16. 一分钟复习

```text
Online MTP：
当前 Policy Rollout Token
→ MTP Future-token CE
→ 通常只更新 Drafter

Rollout K 决定提出几个候选；
Train D 决定训练几个未来深度；
K 与 D 可以解耦。
```

```text
MTP 三臂：
Off → Frozen：纯投机推理收益
Frozen → Online：在线适配收益 - Actor 训练成本
Off → Online：端到端总价值
```

```text
性能归因：
Actor logprob 重算 ≠ Activation Recompute
CE Fusion ≠ LM Head Fusion
长 NCCL Envelope ≠ 全是网络搬运
Profiler 归因 ≠ 正式吞吐
HF 权重恢复 ≠ 完整断点续训
```

## 17. 自测问题

### 问题 1

为什么 Rollout K2/Train D1 是合法而且可能更划算的配置？

期望回答：同一物理 Drafter 可在推理时递归调用两次，但训练只构造一份未来 Token Loss；这样可能保留 K2 的接受长度，同时减少一个 Full Attention/MoE/LM Head/CE/Backward 深度。

### 问题 2

为什么 Frozen→Online 比 Off→Online 更适合衡量在线 MTP 训练本身的价值？

期望回答：两边都启用相同的投机 Rollout K，差分主要剩 MTP 是否在线更新及其训练成本；Off→Online 同时改变了推理和训练两部分。

### 问题 3

为什么冻结 LM Head 权重后，Profiler 里仍可能看到很大的 Projection Backward GEMM？

期望回答：冻结只省 WGRAD；MTP Loss 仍需要对 Projection 输入求 DGRAD，把梯度传回 MTP Hidden State。

### 问题 4

怎样判断一个数秒的 MoE All-to-All 是网络传输慢，还是 Rank Arrival Skew？

期望回答：跨 Rank 对齐 Collective 的进入时间和 Kernel 时长；若早到 Rank 等数秒，而晚到 Rank 真正 Kernel 只有几十毫秒，应优先查晚到 Rank 的上游阻塞。

### 问题 5

为什么加载 HF 权重并把日志 Step 改成旧值，仍不能叫精确续训？

期望回答：Optimizer、Scheduler、RNG、Dataloader Position 等状态没有恢复，数据与优化轨迹会重新开始。

## 18. 与已有知识的联系

- [On-policy 训推、Logprob 与 MTP 性能分析](OnPolicy训推与MTP性能分析_20260806.md)：补充了 Online MTP 三臂与更细粒度训练归因。
- [MTP 与 Context Parallel 训练正确性](MTP与ContextParallel训练正确性_20260817.md)：延伸了 K/D 解耦、Agentic-safe Mask 和词表轴成本。
- [MoE Router Replay 与训推路由一致性](MoERouterReplay与训推路由一致性_20260817.md)：补充了 EP All-to-All 的 Payload/等待拆分。
- [MTP 模型状态流与在线权重事务](../07/MTP模型状态流与在线权重事务_20260729.md)：补充了 HF Warm Start 与完整 Resume 的区别。

## 19. 尚未解决与后续路线

- 用固定 Actor-ready Batch 做 K2/D1 与 K2/D2 的严格 Profiler A/B；
- 量化不同长度下 MTP Full Attention、MoE、Projection、CE、Recompute 的比例；
- 验证 Agentic 多轮/Tool 数据的累计 MTP Mask；
- 将 Accept Length/Rate 按 Policy Version、长度和 Draft Depth 自动记录；
- 结合 Allocator Snapshot 与 CPU Sampling，继续区分 Host Allocation、Allocator Lock/Reclaim 和线程调度。
