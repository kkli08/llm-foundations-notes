# On-policy 训推、Logprob 与 MTP 性能分析

> 日期：2026-08-06
> 来源：当天 Codex 对话；此前与 Gemini 的学习内容转述
> 状态：已整理、已校正边界条件

## 整理记录

- 2026-08-06：整理 Logprob、On-policy/Off-policy、PPO/GRPO，以及 MTP 投机解码性能压测与 Profiler 分析。
- 本文只记录可复用的公开基础理论，不包含工作仓库、内部环境和具体任务安排。
- 用户提供的复习材料保留了主线，但将若干绝对化表述改成了带前提的工程结论。

---

## 0. 今天最终要解决的问题

1. Logprob 具体表达什么，怎样从 Logits 计算出来？
2. 为什么 PPO/GRPO 使用 Logprob，而不是直接使用概率？
3. On-policy 和 Off-policy 的本质区别是什么？
4. On-policy RL 的主要性能问题是否就是 Rollout 推理效率？
5. PPO 和 GRPO 如何把 Reward 变成 Actor 参数更新？
6. 为什么 MTP 在低 Local Batch Size 下通常更容易获得加速，而高 Batch 下可能收益下降？
7. 怎样设计不被预热、输出长度、CUDA 异步和 Profiler 污染的性能实验？

---

## 1. 先给结论

1. **Logprob 是模型为某个确定 Token 分配概率的自然对数**：

   ```text
   Logits → Softmax → Token Probability → log → Token Logprob
   ```

2. 对同一条已生成 Response，PPO/GRPO 比较同一 Token 在 Current 与 Behavior/Old 策略下的 Logprob：

   \[
   r_t
   = \frac{\pi_{\text{current}}(y_t\mid s_t)}
           {\pi_{\text{old}}(y_t\mid s_t)}
   = \exp\left(\log p_{\text{current},t}-\log p_{\text{old},t}\right)
   \]

3. **On-policy/Off-policy 看的是数据由谁生成**，不是 Rollout 和 Trainer 是否部署在同一个进程：

   - On-policy：数据来自当前策略或非常接近当前策略的最新快照；
   - Off-policy：数据可以来自明显不同的旧策略、其他策略或 Replay Buffer。

4. 在 LLM On-policy RL 中，Rollout 经常是主要耗时，因此推理优化通常非常重要；但闭环瓶颈还可能在 Reward/Reference Forward、Actor 训练、权重同步和流水线空泡，不能只优化 Rollout TPS。

5. PPO 用 Critic/Value 估计 Baseline；GRPO 用同一 Prompt 下一组 Response 的平均表现作为 Baseline。二者通常都用 Current/Old Ratio 和 Clip 限制局部策略更新。

6. MTP 的核心不是“免费增加预测”，而是：

   > 用较便宜的 Draft 计算和一次成块 Target Verify，减少昂贵 Target 串行调用次数。

   小 Batch Decode 往往算术强度较低、偏 Memory-bound，MTP 更容易把一次权重访问摊到多个候选 Token 上；大 Batch 已能更充分复用权重、提高计算利用率，MTP 的额外 Draft、Verify、Sampling 和状态管理开销更难隐藏。

7. `BS=1～2`、`BS=16～32`、`1.5×～1.8×`、预热 `1～2 Step` 都只能作为某次实验假设或经验起点，不能写成跨模型、跨硬件的通用阈值。

---

## 2. Logprob 到底是什么

### 2.1 从 Hidden State 到 Logprob

Transformer 最后一层给当前位置输出 Hidden State：

\[
h_t\in\mathbb{R}^{H}
\]

LM Head 把它映射到整个词表：

\[
z_t=W_{\text{vocab}}h_t+b
\]

其中：

- (H) 是 Hidden Size；
- (V) 是 Vocabulary Size；
- (z_t\in\mathbb{R}^{V})；
- (z_{t,j}) 是第 (j) 个候选 Token 的 Logit。

Logit 是还没有归一化的分数，不是概率。经过 Softmax：

\[
p_{t,j}
=\frac{e^{z_{t,j}}}{\sum_{k=1}^{V}e^{z_{t,k}}}
\]

如果当前位置实际选择的 Token 是 (y_t)，它的 Logprob 是：

\[
\log p_t(y_t)
=\log\operatorname{softmax}(z_t)_{y_t}
=z_{t,y_t}-\operatorname{logsumexp}(z_t)
\]

代码通常直接使用 `log_softmax`，而不是先算 `softmax` 再取 `log`，因为前者数值更稳定。

### 2.2 一个三 Token 的数值例子

假设词表只有：

```text
[猫, 狗, 鸟]
```

模型输出 Logits：

```text
[2.0, 1.0, 0.0]
```

Softmax 后约为：

```text
P(猫) = 0.665
P(狗) = 0.245
P(鸟) = 0.090
```

对应 Logprob：

```text
log P(猫) ≈ -0.408
log P(狗) ≈ -1.408
log P(鸟) ≈ -2.408
```

需要形成两个直觉：

- 概率越接近 1，Logprob 越接近 0；
- 概率越小，Logprob 是绝对值更大的负数。

因为概率位于 ((0,1]\)，所以正常 Token Logprob 不大于 0。

### 2.3 为什么经常使用 Logprob

#### 原因一：序列概率的乘法变成加法

自回归模型中，一整段 Response 的联合概率是：

\[
P(y_{1:T}\mid x)
=\prod_{t=1}^{T}P(y_t\mid x,y_{<t})
\]

取对数后：

\[
\log P(y_{1:T}\mid x)
=\sum_{t=1}^{T}\log P(y_t\mid x,y_{<t})
\]

很多很小的概率连续相乘容易数值下溢；Logprob 求和更稳定，也更便于计算。

#### 原因二：Cross-Entropy 本质上就是负 Logprob

对正确 Token (y_t)：

\[
L_{\text{CE},t}=-\log P(y_t\mid s_t)
\]

正确 Token 概率越高，负 Logprob 越小，Loss 越小。

#### 原因三：策略比值可通过 Logprob 差计算

\[
\frac{p_{\text{current}}}{p_{\text{old}}}
=\exp(\log p_{\text{current}}-\log p_{\text{old}})
\]

不需要直接对两个很小的概率做除法。

### 2.4 一段 Response 的 Logprob 不能脱离长度比较

原始 Sequence Logprob 是 Token Logprob 的和。由于每一项通常都小于等于 0：

```text
Response 越长
→ 累加的负数越多
→ 原始 Sequence Logprob 往往越负
```

所以不同长度 Response 不能只按原始 Logprob 和直接判断谁更好。根据目的，可能需要比较：

- 平均 Token Logprob；
- Perplexity；
- 长度归一化 Score；
- 相同 Token 位置上的策略 Logprob。

### 2.5 RL 中的四种 Logprob

对同一个 Prompt、同一条 Response、同一个 Token 位置：

| 名称 | 由谁计算 | 主要用途 |
|---|---|---|
| Behavior/Old Logprob | 真正生成本批数据的 Rollout 策略 | Ratio 分母，说明数据来自谁 |
| Prox Logprob | 本轮局部优化的冻结锚点 | 某些实现中的 PPO Clip 分母 |
| Current Logprob | 当前带梯度 Actor | Ratio 分子，参与 Policy Loss |
| Reference Logprob | 通常冻结的参考策略 | Actor/Reference KL 约束 |

它们评估的是同一批固定 Token，而不是各自重新 Sampling 一条不同 Response。

### 2.6 Sampling 分布与 Logprob 契约

如果 Rollout 使用 Temperature、Top-k 或 Top-p，必须明确保存的 `old_logprob` 指什么：

1. 原始模型 Softmax 分布下的 Logprob；还是
2. Temperature 调整并经过 Top-k/Top-p 截断、重新归一化后的实际采样分布 Logprob。

严格的 Importance Ratio 需要分母对应真实 Behavior Distribution。不同框架可能选择不同算法契约，不能只看到字段名 `old_logprob` 就猜测。还需要核对：

- Tokenizer 与 Chat Template；
- Temperature、Top-k、Top-p；
- EOS/Stop 规则；
- Token Mask；
- 权重版本；
- Logprob 是采样时保存还是训练时重算。

---

## 3. On-policy 和 Off-policy

### 3.1 在 LLM 中怎样映射

| RL 概念 | LLM 对应物 |
|---|---|
| State (s_t) | Prompt + 已生成 Token 前缀 |
| Action (a_t) | 下一个 Token |
| Policy \(\pi_\theta\) | Actor 的 Token 概率分布 |
| Trajectory | 一整条 Response |
| Reward | Verifier、Reward Model 或规则得分 |
| Rollout | 让 Actor/推理引擎真实生成 Response |

定义：

- Behavior Policy：实际生成训练数据的策略；
- Target/Current Policy：当前要优化的策略。

### 3.2 On-policy

```text
Actor v_k
→ 同步到 Rollout
→ Rollout v_k 生成新数据
→ Reward / Advantage
→ 更新 Actor 得到 v_k+1
→ 再同步并重新采样
```

数据通常只做有限轮更新，随后丢弃并重新 Rollout。

### 3.3 Off-policy

```text
Actor v2 / 其他策略 / 历史日志生成数据
→ Replay Buffer
→ 数据被用于训练 Actor v10
```

优势是样本可以反复复用、Rollout 和 Trainer 更容易解耦；代价是需要处理策略分布偏差、数据陈旧、Reward Drift 和 Behavior Logprob 完整性。

### 3.4 PPO/GRPO 为什么有 Old Policy 仍算 On-policy

PPO/GRPO 的一批数据由冻结的 (\pi_{\text{old}}) 生成。优化过程中 (\pi_{\text{current}}) 会逐渐变化，因此二者不会始终完全相同。

它们仍通常归为 On-policy，是因为：

- Old 是当前策略的最新快照；
- 数据只被短期、有限次数复用；
- Ratio/Clip 约束 Current 不要迅速远离 Old；
- 一轮结束后刷新策略并重新采样。

所以 On-policy 并不意味着每次梯度更新前都必须重新生成一条 Response，而是强调数据分布与当前策略保持接近。

### 3.5 两类方法的训推挑战

| 维度 | On-policy | Off-policy |
|---|---|---|
| 数据新鲜度 | 需要严格追踪权重版本和有限 Policy Lag | 允许旧数据，但要管理年龄和版本 |
| 系统吞吐 | Rollout 慢会让 Trainer 等待 | Rollout/Trainer 更易解耦 |
| 样本复用 | 有限，样本效率较低 | 可使用 Replay Buffer 反复复用 |
| 概率校正 | Current/Old 通常较接近 | Importance Weight 可能高方差 |
| 数据契约 | Old Logprob、Tokenizer、Mask、版本必须准确 | 除左侧外还要管理 Reward/格式随时间漂移 |
| 主要失败模式 | Stale Rollout、权重没同步、流水线空泡 | Distribution Shift、Coverage 缺失、陈旧 Reward |

---

## 4. PPO 与 GRPO 的最小完整骨架

### 4.1 PPO

PPO（Proximal Policy Optimization）通常是 On-policy Policy Gradient 算法。典型 LLM PPO 包含：

```text
Actor / Current Policy
Old / Behavior Policy
Critic / Value Model
Reference Policy（LLM RL 中常见，并非通用 PPO 必需角色）
Reward Model / Verifier
```

典型流程：

```text
Old Policy Rollout
→ Reward
→ Critic 估计 Value
→ 计算 Advantage
→ Current/Old Ratio
→ PPO Clip Actor Loss
→ Critic Loss
→ Backward / Optimizer
```

PPO Clip 目标的直觉形式：

\[
L_{\text{PPO}}
=\mathbb{E}\left[
\min\left(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

- (A_t>0)：提高该 Token 的概率；
- (A_t<0)：降低该 Token 的概率；
- Clip：即使更新方向正确，也不继续奖励过大的单轮变化。

需要注意：Clip 是优化目标中的软限制，不是保证所有 Ratio/KL 都严格不越界的硬投影。

### 4.2 GRPO

GRPO（Group Relative Policy Optimization）是 PPO 风格的策略优化方法。它不训练独立 Critic，而是对同一个 Prompt 生成一组 Response，并用组内表现建立 Baseline。

例如同一个 Prompt 生成 4 个答案：

```text
Rewards = [1, 1, 0, 0]
Mean    = 0.5
Std     = 0.5
```

组相对 Advantage：

\[
A_i=\frac{r_i-\operatorname{mean}(r)}
{\operatorname{std}(r)+\varepsilon}
\]

得到近似：

```text
Advantages = [+1, +1, -1, -1]
```

因此：

- 高于同组平均水平的 Response 被提高概率；
- 低于同组平均水平的 Response 被降低概率；
- Actor 更新仍可使用 PPO 风格 Ratio、Clip 和 Reference KL。

GRPO 不是“没有 Baseline”，而是把 Critic Baseline 换成了 Group-relative Baseline。

### 4.3 PPO 与 GRPO 对比

| 维度 | PPO | GRPO |
|---|---|---|
| Advantage 来源 | Return 减 Critic Value | Reward 减组内平均 Reward |
| Critic | 通常需要 | 不需要 |
| 同 Prompt 多次采样 | 不一定 | 通常需要 |
| 额外主要成本 | Critic Forward/Backward/显存 | Group Rollout 成本 |
| 主要不稳定来源 | Critic 偏差、Actor/Critic 联合训练 | 组内 Reward 方差不足、粗粒度 Credit Assignment |

如果同组 Reward 全相同：

```text
[0, 0, 0, 0]
或 [1, 1, 1, 1]
```

组内标准化后几乎没有有效相对 Advantage。因此需要监控：

- Zero-variance Group 比例；
- Reward 均值与标准差；
- 每个 Prompt 的 Group Size；
- 正负样本比例；
- 每个位置或每条 Response 的 Advantage/Mask。

---

## 5. On-policy 的关键是否就是 Rollout 推理优化

### 5.1 结论：经常是，但不是只有它

一个外层 RL Step 的关键路径可以近似拆成：

\[
T_{\text{step}}
\approx
T_{\text{rollout}}
+T_{\text{reward/ref}}
+T_{\text{train}}
+T_{\text{sync}}
-T_{\text{overlap}}
\]

对于长 Response、较大 Group Size 的 GRPO/RLVR：

```text
一个 Prompt
→ 生成 G 条长 Response
→ 大量逐 Token Decode
```

因此 Rollout 经常占据大头，优化推理侧能够直接缩短新鲜轨迹的生产时间。

### 5.2 Rollout 侧值得优化什么

- Continuous Batching 与调度；
- KV Cache/Paged KV 管理；
- Prefix Cache；
- MTP/Speculative Decoding；
- TP/DP/EP 布局；
- CUDA Graph 与 Kernel Fusion；
- 合理的 Local Batch Size；
- Prompt/Response 长度与 Padding 控制；
- Actor、Reference、Reward、Rollout 的共址或资源切分。

### 5.3 不能只追求 Rollout TPS

On-policy 还要求：

```text
快
+ 数据新鲜
+ 策略版本正确
+ Behavior Logprob 正确
+ 下一轮真的使用了新权重
```

几个典型反例：

1. 把队列积得很长，Rollout TPS 提高了，但 Trainer 消费时数据已经落后多个策略版本；
2. 使用更大的 Batch 提高吞吐，却让权重切换等待时间和 Policy Lag 增大；
3. MTP 提高了候选生成量，但接受率低，Target Verify、Reject 和 Cache 管理成本超过收益；
4. Rollout 已很快，但 Actor Backward、Reward Model 或跨节点 Weight Sync 才是主瓶颈。

因此更准确的目标不是单独最大化推理 TPS，而是：

> 在允许的 Policy Lag、训练稳定性和资源预算内，最大化单位时间产生并消费的有效新鲜样本。

---

## 6. Roofline、Local Batch Size 与 MTP 加速

### 6.1 Memory-bound 和 Compute-bound

Roofline 模型关注算术强度：

\[
\text{Arithmetic Intensity}
=\frac{\text{FLOPs}}{\text{Bytes moved}}
\]

- 算术强度低：搬运数据多、计算少，容易 Memory-bound；
- 算术强度高：数据被重复用于更多计算，容易逐渐转向 Compute-bound。

Decode 阶段每轮只为每条活跃序列处理少量 Query Token，而模型权重很大。Local Batch 较小时，Linear 层在物理行为上接近小 (M) 的矩阵乘法，权重复用有限，常表现为显存带宽受限。

Local Batch 增大后，多条序列可以复用已经搬到片上层次的权重，矩阵的 (M) 维增大，算术强度和计算单元利用率通常提升。

但实际瓶颈还受以下因素影响：

- 模型结构和参数规模；
- TP/EP 布局；
- Context Length 和 KV 访问；
- 数据类型、量化；
- GPU 型号和内存带宽；
- Kernel 实现；
- Continuous Batching 的动态形状；
- 通信与调度。

因此不能只用固定的 Batch Size 数字判断 Memory-bound 或 Compute-bound，应该结合 Profiler/Roofline 指标实测。

### 6.2 MTP 为什么更偏爱小 Batch 场景

普通自回归 Target Decode：

```text
每生成 1 个 Token
→ 运行一次完整 Target
→ 再生成下一个 Token
```

MTP/Speculative Decoding：

```text
较便宜的 MTP/Draft 先提出 K 个候选
→ Target 对已知候选成块 Verify
→ 一次 Target 调用可能提交多个 Token
```

如果 Target 的单 Token Decode 主要在等待权重和状态搬运，那么一次 Target Verify 同时评分多个已知候选，可以提高这次权重访问对应的有效计算量，并把 Target 调用成本摊到多个最终 Token 上。

更准确的成本比较是：

\[
T_{\text{spec-step}}
=T_{\text{draft}}
+T_{\text{target-verify}}
+T_{\text{accept/sample}}
+T_{\text{state}}
\]

对比：

\[
N_{\text{emitted}}
\times T_{\text{target-single-decode}}
\]

只有当一次 Speculative Step 平均提交的 Token 足够多，且额外开销足够小，MTP 才真正加速。

### 6.3 高 Batch 下为什么可能收益下降

高 Local Batch 的普通 Decode 已经提高了权重复用与 GPU 利用率。此时再增加：

- MTP Layer/Head Forward；
- 大词表 Logits Projection；
- 多候选 Target Verify；
- Reject/Accept Sampling；
- 临时 KV/递归状态管理；
- Batch Expansion 或调度开销；

这些计算和状态成本更难被空闲算力隐藏，可能使 Speedup 接近 1，甚至慢于关闭 MTP。

这不是说大 Batch 下一定没有收益，而是收益边界要由以下量共同决定：

```text
Local Batch
× Context Length
× Draft 深度/候选数
× Acceptance Length
× Target Verify 成本
× 硬件与并行布局
```

### 6.4 不要把特定数字写成定律

以下说法只能在对应实验条件下成立：

```text
BS=1～2 一定 Memory-bound
BS=16～32 一定 Compute-bound
MTP 一定获得 1.5×～1.8×
某个 35B 模型固定需要多少毫秒搬运权重
```

应把它们改写成可验证假设：

> 预期较低 Local Batch 更容易处于带宽受限区，因此 MTP Speedup 可能更高；随着 Local Batch 增大，预期额外计算逐渐显性化。通过 Local-BS Sweep、Profiler 和端到端数据验证拐点。

---

## 7. 分布式架构与 Local Batch Size

### 7.1 Local BS 与 Global BS

在每个 DP Replica 负载均衡、每个 Replica 使用相同 Local Batch 的简化条件下：

\[
\text{Global Active Batch}
\approx
\text{Local Active Batch per Replica}
\times \text{DP Replicas}
\]

例如：

```text
TP = 2：两张卡共同构成一个模型副本
DP = 8：共有八个这样的模型副本
Local Active BS = 4：每个副本当前处理 4 条活跃序列

Global Active BS ≈ 4 × 8 = 32
总 GPU 数 = TP × DP = 16（暂不考虑其他并行轴）
```

### 7.2 为什么 MTP 性能曲线首先看 Local BS

增加 DP 主要是复制更多模型实例，增加集群横向吞吐；真正改变单个模型副本 GEMM/GEMV 形状、KV 占用和单步执行行为的是该副本当前实际处理的 Token/Sequence 数量。

但是要区分：

- 配置的 `max_num_seqs`：上限；
- 请求侧并发/QPS：输入负载；
- 调度器某一步形成的 Active Batch：实际物理 Batch；
- 一个 Step 的 Batched Tokens：可能同时包含 Decode 和 Chunked Prefill Token。

所以 Local BS 很关键，但不是系统的“唯一物理自变量”。

### 7.3 节点内与跨节点

- Intra-node 通常使用 NVLink/NVSwitch 或 PCIe；
- Inter-node 通常使用 InfiniBand/RoCE 等网络；
- 跨节点通常具有更高延迟，带宽也常低于节点内高速互联，但差距取决于具体硬件拓扑，不能固定说成一个数量级。

实验中需要记录：

- TP/EP 是否跨节点；
- Collective 的消息大小与频率；
- 是否出现 Straggler；
- p95/p99 是否随跨节点增加而恶化；
- 通信是否与计算重叠。

---

## 8. 性能指标应该怎样看

| 指标 | 回答的问题 | 关键注意点 |
|---|---|---|
| Output TPS | 单位时间最终输出多少 Token | 必须统一“最终输出 Token”口径，不能把被拒绝 Draft 算进去 |
| Total Token TPS | 输入与输出 Token 总处理量 | Prefill 比例变化时不等于纯 Decode 能力 |
| TPOT | 首 Token 后平均每个输出 Token 的时间 | 是请求级指标，与全局 TPS 不总是互为倒数 |
| ITL | 相邻两个输出 Token 的时间间隔 | 更能看到流式抖动 |
| TTFT | 从请求到首 Token 的时间 | 主要受排队、Prefill 和调度影响 |
| Acceptance Rate | Draft Token 中被 Target 接受的比例 | 高接受率不自动等于高 Speedup |
| Acceptance Length | 每次 Spec Step 平均接受/输出多少 Token | 往往比单独 Rate 更接近摊销收益 |
| Peak VRAM | 峰值显存 | 需要同时记录模型、KV、临时 Workspace 与 Profiler 影响 |
| p50/p95/p99 | 常态与尾部延迟 | 应与 Mean、样本量、误差条一起报告 |

### 8.1 Speedup 的条件

常用：

\[
\text{Speedup}_{\text{TPS}}
=\frac{\text{Output TPS}_{\text{on}}}
{\text{Output TPS}_{\text{off}}}
\]

或者：

\[
\text{Speedup}_{\text{latency}}
=\frac{\text{Latency}_{\text{off}}}
{\text{Latency}_{\text{on}}}
\]

只有在请求数、有效输出长度、并发模型、计时区间和 Token 统计口径严格一致时，TPS Speedup 与 TPOT 倒数关系才可能近似成立。

在 Continuous Batching 中：

- 每个请求加入/退出 Batch 的时间不同；
- TPOT 是请求级统计；
- Output TPS 是集群级聚合；

因此不能默认两者数学上永远完全互为倒数。

### 8.2 Mean 与 Percentile 都要保留

不能简单“拒绝使用 Mean”。更合理的是：

```text
吞吐：Mean/Aggregate TPS + 稳态区间
延迟：Mean + p50 + p95/p99
稳定性：标准差/误差条 + 重复轮次 + 样本量
```

- p50 表示典型请求；
- p95/p99 表示尾部；
- Mean 对总资源消耗和总体平均体验仍有意义；
- 极少样本的 p95/p99 没有可靠统计意义。

---

## 9. Benchmark 防污染原则

### 9.1 预热到稳态，而不是固定丢弃 1～2 Step

预热要排除：

- 模型和权重加载；
- CUDA Context/cuBLAS 初始化；
- NCCL 建链；
- Torch Compile/JIT；
- CUDA Graph Capture；
- 内存池与 KV Cache 初始化；
- Prefix Cache 的冷热状态差异。

`1～2 Step` 可以作为 Fast PoC 起点，但正式测试应观察延迟是否已经稳定，或使用固定、足够的 Warm-up 区间。

### 9.2 固定有效输出长度

如果要比较固定 Decode 工作量，可以配置：

```text
min_tokens = max_tokens = N
```

或者使用框架提供的 `ignore_eos`/等价机制。

但仍需核实：

- Stop String/Stop Token 是否还会终止；
- 请求失败是否被排除；
- 最终输出 Token 数是否真的一致；
- MTP 被拒绝的 Draft Token 是否没有混入 Output TPS。

最终以日志中实际 `generated_tokens` 为准，而不是只相信配置。

### 9.3 CUDA 异步计时

CUDA Kernel 默认异步提交。如果直接：

```python
start = time.time()
run_cuda_work()
end = time.time()
```

CPU 可能只统计到 Kernel Launch 时间。

测 GPU 区间可选择：

- 在打点边界使用 `torch.cuda.synchronize()`；
- 使用 `torch.cuda.Event`；
- 使用 `torch.utils.benchmark` 处理预热和同步。

对于多 GPU 端到端测试，还要根据想测量的边界决定是否需要 Rank Barrier。过度插入同步也会改变真实流水线，因此正式 Serving 吞吐更适合用完整请求的端到端 Wall Clock，同时另做 Kernel 微观计时。

### 9.4 Benchmark 与 Profiler 分开

Profiler 会引入：

- CPU 事件记录；
- CUDA Activity 收集；
- Shape/Stack/Memory 记录；
- Trace 写出；

因此：

```text
Run A：关闭 Profiler，获取纯净端到端 TPS/Latency
Run B：开启 Profiler，定位时间花在哪里
```

Run B 主要用于相对归因，不应直接冒充 Run A 的正式吞吐结果。

### 9.5 Off/On 对照至少固定什么

```text
模型 Checkpoint 与精度
硬件和拓扑
TP/DP/EP
Prompt 集合与输入长度
实际输出长度
Sampling 参数和 Seed 策略
Local Active Batch / 请求到达模型
CUDA Graph/Compile 状态
Warm-up 与测量窗口
软件版本
```

MTP Off/On 之间原则上只改变 MTP/Speculative Decoding 相关开关。

---

## 10. PyTorch Profiler + Perfetto 的微观分析

### 10.1 四个阶段与两类核心开销

一次 MTP Decode Iteration 最好拆成四段：

```text
MTP Draft Forward
→ Draft Token Sampling
→ Target Verify
→ Acceptance / Rejection
```

其中负责人常关心的两类核心开销是：

#### A. Draft/Sample 新增开销

关注：

- MTP Layer/Head 的额外耗时；
- 大 Vocabulary LM Head Projection；
- 从 Draft Logits 选出候选 Token 的 Sampling；
- Draft 深度增加后 Tensor Shape 如何变化。

#### B. Target Verify 新增开销

关注：

- Target 成块 Verify 的耗时；
- 候选数增加后 Tensor Shape 如何变化；
- 单次 Verify Duration 与每次最终提交 Token 数。

不要只看“Verify 比单 Token Forward 更慢”，而要看：

\[
\text{Target Cost per Emitted Token}
=\frac{T_{\text{target verify}}}
{\text{本次最终提交 Token 数}}
\]

Acceptance/Rejection 应与 Target Verify 分开统计。Verify 负责计算候选位置的 Target 结果；Acceptance/Rejection 负责比较候选、选择最长有效前缀，并提交或丢弃相应状态。

状态管理还需关注：

- Rejection Sampling / Greedy Verify；
- Draft 候选校验；
- Accepted Prefix 计算；
- Bonus Token；
- KV Cache 或递归状态的 Commit/Discard；
- CPU 调度、GPU Kernel Launch 和跨 Rank 同步。

实际算子名称依赖框架版本，不能假定一定叫 `speculative_sampling` 或 `verify_and_sample`。最好在关键区间增加 `record_function`/NVTX 标记。

### 10.2 CUDA Graph 下的注意点

`enforce_eager=False` 是否等价于启用某种 CUDA Graph 模式取决于具体推理框架和版本。正式笔记应记录实际生效的 Graph Mode，而不是只记录配置值。

对于 CUDA Graph Replay：

- 先完成 Capture；
- Replay 若干次预热；
- 再采集稳定 Trace；
- 必要时使用 Graph Annotation/NVTX 恢复业务区间语义；
- 对照一次 Eager Trace，避免 Graph Replay 把算子归因压缩得难以理解。

### 10.3 CUDA Graph、Kernel、Bucket 与 Profiler 的分工

四个概念不能混在一起：

| 对象 | 作用 | 不负责什么 |
|---|---|---|
| CUDA Kernel | 在 GPU 上执行一段张量计算 | Token 不是 Kernel 的固定执行单位 |
| CUDA Graph | 录制固定形状下的一串 Kernel 与依赖，随后低开销重放 | 不负责监测、计时或理解 Draft/Verify 语义 |
| Capture Bucket | 声明预录制的总 Token Slot 形状 | 不表示“第几个 Token”，也不天然表示 Draft/Verify 阶段 |
| Profiler | 记录 CPU Op、CUDA Kernel、Graph Replay 与耗时 | 不会自动知道某组 Kernel 属于哪个业务阶段 |

大模型一次 Forward 会触发多类 Kernel，例如 GEMM、Attention/GDN、Norm、MoE Router/Expert、Logits Projection 和 Sampling。一个 Kernel 通常处理包含多个 Token Slot、Head 和 Hidden Dimension 的张量；不能理解成“一个 Token 对应一个 Kernel”。

以单请求、投机深度 `K=2` 为例，Target Verify 的均匀 Decode 形状通常是：

```text
BS = 1
K = 2
→ K + 1 = 3 个 Token Slots
→ 一次 [3, hidden_size] 级别的 Forward
→ 多个 CUDA Kernels
```

此时只有 Bucket 1 不代表运行时会把 3 个 Token 拆成三次 Graph-1。框架通常先把 Spec Decode Capture Size 对齐到 `K+1` 的倍数；没有合法形状时会拒绝配置或退回其他执行模式。只有 Bucket 6 时，3 个实际 Slot 可能 Padding 到 6 后重放 Graph-6，但不会因为实际输入是 3 就自动生成 Graph-3。

Profiler 会自动抓全执行事件，但阶段归因仍需要语义边界。推荐顺序是：

1. 先使用框架已有的 Range、Module Path 和 Graph Shape；
2. 如果仍不能区分，在调用边界增加最小 `record_function`/NVTX 标签；
3. 使用 `MTP_DRAFT_FORWARD`、`MTP_DRAFT_SAMPLE`、`TARGET_VERIFY_FORWARD`、`ACCEPT_REJECT` 四段；
4. 不逐个 Kernel 猜业务阶段，也不直接把 Graph-1/Graph-3 等同于 Draft/Verify。

两类增量开销可以按下面口径计算：

\[
T_{\text{verifier-delta}}
=T_{\text{target-verify,on}}
-T_{\text{target-decode,off}}
\]

\[
T_{\text{draft/sample}}
=T_{\text{draft-forward}}
+T_{\text{draft-sampling}}
\]

Acceptance/Rejection 与 Cache/State Commit 单独列出，避免把不同职责混成一个数字。

### 10.4 推荐归因表

| 区间 | MTP Off | MTP On | Delta | 每最终 Token Delta | 解释 |
|---|---:|---:|---:|---:|---|
| Draft/MTP Head | 0 |  |  |  | 新增成本 |
| Target Forward/Verify |  |  |  |  | 成块 Verify |
| Sampling/Accept |  |  |  |  | 拒绝采样与候选处理 |
| Cache/State |  |  |  |  | Commit/Discard |
| Communication |  |  |  |  | TP/EP/DP Collective |
| CPU Scheduling |  |  |  |  | Engine/Scheduler |

---

## 11. Fast PoC 实验设计

### 11.1 第一阶段：端到端干净测速

假设资源允许，固定：

```text
DP = 8
TP = 2
MTP K = 1
Local BS ∈ {1, 2, 4, 16}
MTP ∈ {Off, On}
```

共 8 个核心实验单元。

这里必须把 `K=1` 写清楚到底表示：

- 一个 MTP Draft Depth；
- 一个 Speculative Token；还是

- 框架中的其他参数。

不同实现的 `K` 命名可能不同。

每个单元建议：

1. 完成 Warm-up；
2. 至少执行多轮有效测量；
3. 记录实际 Active Batch 与 Token 数；
4. 汇总 Mean、p50、p95 和重复轮次误差；
5. 保存原始日志和结构化 CSV。

### 11.2 第二阶段：单独抓 Profiler

可以优先选择：

```text
Local BS = 1
MTP Off / On
```

原因是预期该点最容易看到 MTP 的带宽摊销收益与新增开销。

但如果端到端曲线显示拐点出现在 `BS=4` 或 `BS=16`，还应该在拐点附近补抓 Trace，回答：

> 为什么收益在这里开始消失？

### 11.3 推荐宽表

| Local BS | MTP | Actual Output Tokens | Output TPS | TTFT Mean/p50/p95 | TPOT Mean/p50/p95 | Acceptance Rate | Acceptance Length | Peak VRAM | Target Verify ms | Draft ms | Sample/State ms |
|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Off |  |  |  |  | N/A | N/A |  |  | 0 |  |
| 1 | On |  |  |  |  |  |  |  |  |  |  |

### 11.4 最有解释力的图

建议至少画三张：

1. `Local BS → Output TPS`，MTP Off/On 两条线；
2. `Local BS → Speedup`；
3. `Local BS → Acceptance Length、Draft/Verify Cost per Emitted Token`。

这样可以把“有没有加速”和“为什么加速/为什么失效”连接起来。

---

## 12. 常见误解与修正

| 容易误解为 | 更准确的理解 |
|---|---|
| Logprob 是模型输出的原始分数 | 原始分数是 Logit；Logprob 是 Log-softmax 后选中 Token 的对数概率 |
| Logprob 越负越好 | 越接近 0 表示该 Token 概率越高；好坏还取决于目标和上下文 |
| Sequence Logprob 可以直接跨长度比较 | 原始和天然偏向短序列，需要长度口径 |
| Rollout 和 Trainer 分开就是 Off-policy | 分类看数据策略版本，不看部署进程 |
| PPO 有 Old Policy，所以是 Off-policy | Old 是最新冻结快照，数据短期复用并受 Ratio/Clip 约束，通常仍归 On-policy |
| On-policy 只需要优化推理 TPS | 还需要数据新鲜度、准确 Logprob、权重同步和训练吞吐 |
| 小 Batch 一定 Memory-bound，大 Batch 一定 Compute-bound | 这是常见趋势，具体拐点必须按模型、硬件、长度和 Kernel 实测 |
| MTP 用的是“免费算力” | MTP 有真实额外成本，只是在某些带宽受限场景中成本可被摊薄或隐藏 |
| Acceptance Rate 高就一定快 | 还要看 Draft、Verify、Sampling、状态管理成本和 Acceptance Length |
| Global BS 决定单副本矩阵形状 | 单副本实际 Active Local Batch 更直接，但序列长度和并行布局等也会影响 |
| TPS Speedup 永远等于 TPOT 倒数 Speedup | 只在严格对齐且统计口径简单时近似成立 |
| 延迟统计应该拒绝 Mean | Mean、p50、p95/p99 回答不同问题，应一起报告 |
| 固定预热 1～2 Step 就一定进入稳态 | Compile、Graph Capture、NCCL 等可能需要更长预热，应验证稳定性 |
| Profiler Trace 中的 TPS 可直接作为正式结果 | Profiler 有 Observer Overhead，应和纯净测速分开 |

---

## 13. 一分钟复习

1. Logprob：`log_softmax(logits)[selected_token]`；Sequence Logprob 是各条件 Token Logprob 的和。
2. PPO Ratio：`exp(current_logp - old/prox_logp)`，比较当前策略与生成/锚点策略对同一 Token 的概率变化。
3. On-policy 使用当前或最新策略的新鲜数据；Off-policy 可以复用明显不同策略的历史数据。
4. PPO 用 Critic 建立 Advantage Baseline；GRPO 用同 Prompt 的组内 Reward 建立 Baseline。
5. LLM On-policy RL 经常由 Rollout 主导耗时，但还必须优化 Reward/Reference、Actor Train、Weight Sync 和流水线重叠。
6. 小 Local Batch Decode 常更偏带宽受限，MTP 更容易摊薄 Target 调用；大 Batch 下额外计算更容易显性化。
7. MTP 加速必须联看 Output TPS、TPOT、Acceptance Length、Draft/Verify/State 成本，不能只看 Acceptance Rate。
8. Benchmark 先预热到稳态、固定实际工作量、正确处理 CUDA 异步，并把正式测速与 Profiler 分开。
9. CUDA Graph 负责固定形状的 Kernel 录制与重放；Profiler 负责观测，Bucket 只描述 Token Slot 形状，三者都不会自动表达 Draft/Verify 业务语义。

---

## 14. 自测问题

### 问题 1：Logit、Probability、Logprob 是怎样连接的？

期望回答：

```text
Logit 是 LM Head 的未归一化分数；
Softmax 把全部 Logit 变成和为 1 的概率；
选中 Token 的概率取自然对数就是 Logprob；
实现中通常直接使用 log_softmax。
```

### 问题 2：为什么 Ratio 使用 Logprob 差？

期望回答：

```text
current_prob / old_prob
= exp(log(current_prob) - log(old_prob))。
Log 空间数值更稳定，也方便处理自回归序列的概率乘积。
```

### 问题 3：Rollout v0、Actor v2 的数据一定不能训练吗？

期望回答：

```text
它已经具有明显 Off-policy/Stale 性质。
是否可用取决于算法是否显式支持策略偏差校正、允许的 Policy Lag、
Behavior Logprob 是否准确以及 Ratio/有效样本量是否可控；
不能把它当作严格新鲜的 On-policy 数据静默使用。
```

### 问题 4：为什么 MTP 在 Local BS=1 可能快，在 Local BS=16 反而变慢？

期望回答：

```text
低 Batch Target Decode 可能主要受权重/状态搬运限制，
成块 Verify 能把一次 Target 调用摊给多个 Token；
高 Batch 普通 Decode 已提高权重复用和计算利用率，
MTP 的 Draft、额外词表投影、Verify、Sampling 和状态管理成本更难隐藏。
具体拐点仍需实测。
```

### 问题 5：为什么不能只看 Acceptance Rate？

期望回答：

```text
接受率没有包含 Draft、Target Verify、Sampling、Cache/State 和通信成本，
也不完整表达一次 Target 调用最终提交多少 Token。
需要和 Acceptance Length、每最终 Token 成本、TPS/TPOT 一起看。
```

### 问题 6：On-policy RL 的系统目标是什么？

期望回答：

```text
不是单独最大化 Rollout TPS，
而是在允许的 Policy Lag、训练稳定性和资源预算内，
最大化单位时间产生并消费的有效新鲜样本。
```

### 问题 7：CUDA Graph、Bucket 与 Profiler 分别负责什么？

期望回答：

```text
CUDA Graph 录制并重放固定形状的一串 GPU Kernel；
Bucket 指定预捕获的总 Token Slot 形状；
Profiler 记录执行事件与耗时。
Profiler 能抓到 Kernel，但不会天然理解 Draft、Verify 或 Accept，
需要依靠已有调用范围、模块路径或最小语义标签完成归因。
```

---

## 15. 后续学习路线

- PPO：GAE、Critic Loss、Entropy Bonus、Clip Fraction 与 Approx KL；
- GRPO：Outcome/Process Supervision、Token-level Advantage、长度偏差和不同实现变体；
- On-policy 系统：异步 Rollout、Policy Lag 上限、Stale Sample 丢弃和流水线建模；
- MTP：Acceptance Length 与理论 Speedup 模型；
- 性能分析：Nsight Systems/Nsight Compute、Roofline 点位和跨 Rank Critical Path；
- Batch：Continuous Batching 下 Active Sequences、Batched Tokens 与 Local BS 的准确口径。

---

## 16. 参考材料

- [PPO 原始论文：Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up：PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [DeepSeekMath：GRPO](https://arxiv.org/abs/2402.03300)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/pdf/GPU-Performance-Background-User-Guide.pdf)
- [PyTorch CUDA Semantics：异步执行与精确计时](https://docs.pytorch.org/docs/main/notes/cuda.html)
- [vLLM 0.19.1：Spec Decode CUDA Graph Capture Size 对齐](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/config/compilation.py#L1132-L1173)
- [vLLM 0.19.1：均匀 Decode 的 K+1 Token Shape](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/worker/gpu_model_runner.py#L725)
- [PyTorch Benchmark Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch CUDA Graph Kernel Annotations and Profiling](https://docs.pytorch.org/tutorials/advanced/cuda_graph_annotations_tutorial.html)
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
