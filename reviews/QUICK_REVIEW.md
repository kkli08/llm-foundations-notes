# 快速复习

> 最后更新：2026-08-26

## 开始新学习前

用 5～10 分钟完成：

1. 阅读本页“当前核心心智模型”；
2. 阅读最近一篇正式笔记的“一分钟复习卡”；
3. 不看答案回答最近一篇笔记的自测问题；
4. 答错的内容加入当天 Inbox。

## 当前核心心智模型

```text
Online MTP：
当前 Policy Rollout Token
→ Future-token CE/SFT
→ 通常只更新 MTP Drafter

Rollout K：推理提出几个候选
Train D：训练几个未来深度
K 与 D 可以解耦。
```

```text
三臂对照：
Off → Frozen：纯投机推理收益
Frozen → Online：在线适配收益与 Actor 成本
Off → Online：端到端总价值

Actor recompute_logprob ≠ Activation Recompute；
CE Fusion ≠ LM Head Projection Fusion。
```

```text
性能归因：
CPU Envelope ≠ 纯 CPU 计算
长 NCCL Range ≠ 全是网络传输
Torch-Compiled Region ≠ 正在编译
Profiler Trace ≠ 生产吞吐
HF 权重恢复 ≠ 完整断点续训
```

```text
MTP + Pipeline Parallel：
PP 切 Layer，MTP 通常属于最后 Stage
→ Layout 物化 MTP
→ Owner 构造 Supervision/Loss
→ Loss 只按 DP×CP Token 归一化
→ Metrics 从最后 Stage 回传 PP0
→ Checkpoint/Weight Sync 合并 Exact Set

Logical MTP Depth 决定计算次数；
Physical MTP Layers 决定参数、显存和 Checkpoint。
```

```text
Online RL Cohort：
同一 Prompt 的 G 条 Session 往往要整组就绪
→ Global Concurrency 分到 R 个 Backend
→ 若 Cohort 绑定单 Backend，要求 C_backend >= G
→ Cohort Ready 时间近似由最慢 Session 决定

Reservation Timeout 只能容忍正常排队；
不能修复物理容量不足、Session 泄漏或吞吐过低。
```

```text
推理优化的证据链：
请求配置
→ 参数被解析
→ 运行时实际后端/执行模式
→ 固定负载、预热后的端到端结果

请求值不等于运行事实；
局部 profiler 指标用于解释，端到端 TPOT/TPS 用于验收。
```

```text
QK Norm：稳定每个 Attention Head 的 Q/K 尺度
RoPE：按 Token Position 旋转 Q/K
Attention：当前 Q 读取历史 K/V
LM Head：产生词表 Logits
Local Argmax：TP 每卡先选本地最大，只聚合候选

Fusion 能减少小 Kernel 和中间读写；
不能消除 Full Attention 的历史 KV 扫描，
也不能消除 LM Head GEMM 与自回归依赖。
```

```text
MTP + Context Parallel：
完整序列语义
→ 各 Depth 的 Future Label / Embedding / Mask
→ Packed Boundary 与 Padding
→ CP-aware Shift / 边界通信
→ DP×CP numerator / valid-token count
→ Forward + Recompute + Backward + Optimizer

Logical Prediction Depth K ≠ Physical MTP Layer Count；
CP 切序列 S，不自动切词表 V。
```

```text
MoE Online Router Replay：
Rollout Top-K logical Expert IDs
→ Causal Shift + Token/Layer 对齐
→ Trajectory / Ray
→ Actor 当前 Router Scores
→ Gather Forced IDs
→ 当前权重 + 固定离散路径
→ Router Gradient

只传 IDs，不传旧权重；
同模式重复稳定是硬门，跨 Batch 自然 Route Drift 是诊断。
```

```text
文本
→ Tokenizer
→ Token IDs
→ Embedding / Hidden States
→ 多层 Transformer
   ├── Attention：Q 匹配 K，按权重读取 V
   └── FFN：加工每个 token 的当前表示
→ LM Head
→ Logits
→ Sampling
→ 一个新 Token
```

```text
Prefill：
并行处理 Prompt，并保存每层 Prompt K/V。

Decode：
每个新 token 都经过全部 Transformer 层；
当前 token 每层计算 Q/K/V；
历史 K/V 从 KV Cache 读取；
完成全部层后才产生下一个 token。
```

```text
预训练：海量文本 → Next-token Loss → 通用参数
SFT：Prompt—Response → 监督 Loss → 指令行为
RL 后训练：
Prompt → Rollout → Reward → Advantage
→ Actor Backward / Optimizer
→ 权重同步 → 新一轮 Rollout
```

```text
同一条 Response token：
Old Logprob       → 数据生成时的行为策略概率
Prox Logprob      → 本轮更新开始时的 Actor 锚点
Current Logprob   → 当前带梯度 Actor 的训练概率
Reference Logprob → 冻结参考策略的概率
ratio = exp(Current - Prox)
正 Advantage 推高概率，负 Advantage 压低概率
```

```text
PPO Clip：限制 Current 相对 Prox 的单轮更新幅度
KL 约束：限制 Actor 相对 Reference 的长期漂移
Checkpoint：保存或恢复训练状态
```

```text
固定 Reference 示例：
Reference W0
Rollout W5 → Response + Old Logprob
Actor W5 → Current Logprob → Loss → Backward/Optimizer → W6
W6 Weight Sync → 下一批 Rollout

Current Logprob 先进入 Loss，随后才得到 W6；
Reference 通常是 RL 起点策略，不一定是原始预训练模型。
```

```text
判断 RLVR 是否改善：
Held-out 任务效果
+ Actor/Reference KL
+ Ratio、Clip Fraction、Entropy、Gradient、NaN/Inf

训练 Reward 或 Policy Loss 单独变化都不能证明能力提升。
```

```text
MTP 训练：
完整序列 [A, B, C, D, E]
→ NTP Label 向未来移动 1 位
→ MTP Label 向未来移动更多 Offset
→ 各路径 Cross-Entropy
→ 加权 Total Loss
→ Backward 计算并累积 Gradient
→ Optimizer Step 修改参数

Cross-Entropy 关注 -log P(正确 Token)，
不是预测 Token ID 与正确 Token ID 的数值距离。
```

```text
推理若依赖训练后的 MTP 模块：
参数参与 Loss
→ 获得 Gradient
→ 进入 Optimizer
→ Optimizer Step 更新
├── 持久化：Checkpoint 保存
└── 在线推理：权重映射 → Weight Sync
              → Rollout 使用新版本产生 Draft

在线同步不要求先落盘 Checkpoint；
Optimizer State 用于续训，不发送给 Rollout。
```

```text
MTP 推理：
当前正式状态
→ Draft 多个候选 token
→ Target 用 Causal Mask 成块 Verify 已知候选
→ 提交从开头起的连续有效前缀
→ 拒绝后缀的临时状态逻辑失效

未知正式 Token 的因果依赖没有消失；
变化是一次 Target 块评分替代多轮 Target 串行 Decode。

常规独立 Draft/Target：
Target Verify 计算自己的 KV/State，
不要假设把 Draft KV 直接搬进 Target Cache。

一种递归式原生 MTP Draft 仍可按 Token 串行，
但每步只执行少量 MTP Layer；成本要比较：
k × 小型 MTP + 1 × Target Block Verify
和 k × 完整 Target 单 Token Decode。
“短块 Prefill-like”只描述候选已知后的 Verify。
```

```text
训练部署分层：
本地入口
→ 资源平台：机器、GPU、容器、存储
→ Ray：Head/Worker、资源与进程调度
→ 训练编排层：Rollout、Reward、Update、Sync
├── Megatron：Forward / Backward / Optimizer
└── vLLM：Prefill / Decode / Rollout

平台 RUNNING ≠ Ray Ready ≠ 模型加载 ≠ Baseline 跑通。
```

```text
一个外层 RLVR Step：
一批 Prompt
→ 每个 Prompt 生成多条 Response
→ Reward / Reference / Actor Logprob
→ Group Advantage
→ Policy Loss / Backward / Optimizer
→ Actor 新版本
→ Weight Sync 给 Rollout

1-Step 证明“产生并同步新权重”；
2-Step 才继续证明“下一轮使用了新权重”。
```

```text
Qwen3 MoE/MTP 最小架构：
MoE = Router 在每层为每个 Token 选择 Top-K Expert FFN
TP = 拆矩阵，EP = 拆 Expert，PP = 拆 Layer

MTP 端到端支持：
模型构建/加载
→ Future Label 与 MTP Loss
→ Checkpoint 保存恢复
→ Megatron 到 vLLM 参数转换/同步
→ Draft/Verify、Acceptance 与吞吐验证

Load MTP、Train MTP、Rollout MTP 是三个相关但独立的语义。
```

```text
一层 Pre-Norm Transformer：
y = x + Attention(Norm(x))
z = y + MLP或MoE(Norm(y))

Attention 跨 Token 汇聚信息；
MLP/FFN 加工每个 Token 的当前表示；
Residual 保留原表示并叠加子层修正。

MoE Top-K：
Hidden State → Router Scores → Top-K Expert MLP
→ 选中 Expert 输出加权聚合 → Residual

一层只输出新的 Hidden State；
所有层结束后，LM Head 才产生词表 Logits。
```

```text
MTP 构建状态：
Off：不构建
Load：Checkpoint 有兼容 MTP，严格加载
Init：Checkpoint 无 MTP，只新增并初始化预期模块

运行用途再独立决定：Train MTP? / Rollout MTP?

Model Adapter 负责模型特定构建、加载/初始化和参数映射；
通用训练路径负责 Label/Mask/Loss；
Megatron 负责分布式 Forward/Backward/Optimizer；
vLLM 负责 Draft/Verify。
```

```text
Qwen3-MoE 一层 Native MTP：
h_t ─ hnorm ─┐
              ├─ concat[2H] → Projection[H]
E(x_t+1)─enorm┘
→ 一层 Qwen3-MoE MTP Transformer Layer
→ Final Norm → 共享 LM Head → x_t+2 Logits

两路输入不是直接相加；
MTP 专属参数需要初始化/加载和训练；
默认 detach 只阻断 MTP Loss 到 Backbone 的梯度；
训练与推理必须使用同构模块和同版本参数。
```

```text
MTP 模型状态流：
Optimizer Step 后的 Backbone + MTP
├── DCP：模型 + Optimizer + Scheduler + RNG，用于续训
├── HF Export：Effective Config + Weights + Index，用于重建
└── Online Sync：推理权重 + Manifest + Version，用于更新 Rollout

参数完整性：
expected = converted = received = applied

在线事务：
Prepare → Pause → Transfer → Validate → Commit → Resume

失败时不 Commit、不 Resume；
Fail-closed 保证请求看不到混合权重，不等于 Tensor 自动回滚。

Structure Hash 检查模型结构；
Value Checksum 检查参数数值。
```

```text
RL 训练数据链：
Trajectories
→ Reward / Advantage / Masks
→ 按有效 Token 做 DP 数据分配
→ Padding 或 Segment-aware Packing
→ input_ids [B,S]
→ Microbatches
→ 全局有效 Token numerator/count
→ Megatron Forward/Backward/Optimizer

Ray Head 管集群，不等于训练数据协调角色。
Packed Tensor 的物理相邻不等于同一逻辑序列；
MTP 多步 Label 不能跨 Segment 全局 roll。
```

```text
三种 Scale：
λ_mtp：算法超参数，改变 MTP 辅助目标权重
Token/Microbatch Weight：动态长度下保证全局 Token 平均
Optimizer Loss Scale：低精度数值稳定，更新前还原

MTP Block 的 Dense/MoE 是训练模型结构；
Draft/Verify 是推理职责；
vLLM proposer 必须与 checkpoint 同构。
```

```text
Logprob：
logp(token) = log_softmax(logits)[token]

Sequence Logprob = 各条件 Token Logprob 之和
Cross-Entropy(token) = -logp(正确 token)

PPO/GRPO Ratio：
ratio = exp(current_logp - old/prox_logp)
```

```text
On-policy：当前/最新策略生成新鲜数据，有限复用后重新 Rollout
Off-policy：历史/其他策略数据可进入 Replay Buffer 反复使用

部署分离不决定 On/Off-policy；
真正关键的是 Behavior Policy 与 Current Policy 的版本和分布差距。

PPO：Critic Value 作为 Advantage Baseline
GRPO：同 Prompt 的 Group Reward 作为 Advantage Baseline
二者通常都使用 Current/Old Ratio 与 Clip。
```

```text
On-policy RL Step 时间：
Rollout + Reward/Reference + Actor Train + Weight Sync - Overlap

长 Response / 大 Group Size 时 Rollout 经常占大头，
但目标不是只最大化推理 TPS，而是：
在可接受 Policy Lag 下最大化有效新鲜样本吞吐。
```

```text
MTP 性能直觉：
低 Local BS Decode 常更偏 Memory-bound
→ 成块 Verify 更容易摊薄 Target 权重访问

高 Local BS 已提高权重复用/计算利用率
→ Draft、Verify、Sampling、状态管理成本更难隐藏

是否加速必须联看：
Output TPS + TPOT + Acceptance Length
+ Draft/Verify/Sampling/State Cost
```

## 当前必会解释

- Rollout K 与 Train Depth D 为什么可以不同；
- Off、Frozen、Online 三臂分别隔离什么变量；
- Online MTP 为什么是 RL 旁边的辅助 Token CE/SFT，而不是另一种 RL；
- `detach_encoder` 为什么不消除 MTP Forward、DGRAD 和 Recompute；
- Actor Logprob Recompute 与 Activation Recompute 的区别；
- CE Fusion 为什么不自动消除全词表 Projection；
- 为什么冻结 LM Head 后仍可能存在 Projection DGRAD；
- 怎样区分 MoE Payload、Expert Imbalance 与 Rank Arrival Skew；
- CUDA Memory Snapshot 没有产出时，为什么不能直接断言 OOM 或 Dump API 失败；
- Acceptance Length 与 Acceptance Rate 的口径差异；
- HF Weight Warm Start 与完整断点续训的区别；
- Dense、MoE 和 baseline 分别是什么；
- Transformer 是模型架构，训练和推理都会使用；
- Hidden State、Q/K/V、FFN 分别是什么；
- 为什么历史 K/V 值得缓存，历史 Q 通常不缓存；
- 为什么一个 token 要经过全部层才会产生下一个 token；
- MHA、GQA、MQA 如何影响 KV Cache；
- Prefill 与 Decode 的区别；
- PagedAttention 为什么类似虚拟内存，但不等于模型 Attention；
- 为什么 8K 到 32K 会导致 TTFT 增大、KV Cache 变大、并发下降。
- 预训练、SFT、RL 后训练的目标与关系；
- Reward、Advantage、Reference 分别解决什么问题；
- Old、Prox、Current、Reference Logprob 为什么必须对应同一条 Response token；
- `ratio = exp(current_logp - prox_logp)` 表示什么；
- Advantage 如何通过 Policy Loss、Backward 和 Optimizer 改变 token 概率；
- PPO Clip 与 KL 约束分别限制哪两种偏移；
- Trajectory 为什么需要 Token、Logprob、Mask、Reward 和权重版本；
- 模型家族、Checkpoint 与 Actor/Reference/Rollout 三类运行角色如何区分；
- 在 W0/W5/W6 示例中 Old、Reference、Current Logprob 的版本与因果时序；
- 为什么用于得到 W6 的 Current Logprob 必须在参数更新前计算；
- 为什么 Reference 通常对应 RL 起点策略，而不一定是原始预训练模型；
- 为什么判断 RLVR 是否改善必须联看 Held-out 任务效果、KL 与训练健康；
- 为什么 Causal Mask 阻止模型偷看未来，但训练程序仍能用未来 Token 作 Label；
- NTP 与不同未来 Offset 的 MTP Label 怎样由同一 Token 序列移位得到；
- 为什么 Cross-Entropy 不是 Token ID 之间的数值距离；
- Forward、Loss、Backward 和 Optimizer 分别负责什么；
- 为什么 `loss.backward()` 后、`optimizer.step()` 前权重通常还没变化；
- 为什么 MTP 参数有 `.grad` 但不在 Optimizer 中仍不会更新；
- 推理若使用 MTP Draft，相关参数必须怎样贯穿 Checkpoint、映射与 Weight Sync；
- 为什么组内 Reward 全相同时，相对 Advantage 往往很弱；
- 算法稳定性指标和系统性能指标有什么区别；
- 二元 Reward 表示什么，为什么它不是所有任务的固定评分方式；
- KL 约束为什么不是 Checkpoint 回滚；
- Actor 与 Rollout Engine 为什么分工、何时同步权重；
- MTP 的训练侧和推理侧为什么相关但不相同；
- Draft Reject 为什么必须回滚 Cache 或其他状态；
- 自回归 Decode 仍然串行，为什么已知 Draft 可以被 Target 成块 Verify；
- 为什么 Verify 是按 Target 分布接受候选，而不是对照训练 Label；
- 为什么单链 Draft 的首个 Reject 会使后续候选失效；
- 为什么 Target 通常需要自己的 KV/State，不能泛化成直接搬运 Draft KV；
- 为什么逻辑提交/回滚不等于一定发生显存复制或删除；
- MTP 加速为什么必须联看接受长度、Draft/Verify 成本和 Tokens/Target Forward；
- 为什么递归式 MTP Draft 即使串行，仍可能比重复运行完整 Target 便宜；
- 为什么 MTP 上线后必须按 Batch/QPS、长度和 Sampling 做 off/on 分桶实测；
- DP、TP、PP、EP、CP 分别切分什么。
- 资源平台、Ray、训练编排层、Megatron 与 vLLM 分别负责什么；
- 为什么平台任务 `RUNNING`、Ray Ready、模型加载和 Baseline 跑通是四级不同证据；
- Baseline、Smoke、模型效果实验和系统性能实验怎样区分；
- 为什么一个外层 RL Step 可以包含多条 Trajectory 和多次 Forward/Backward；
- 为什么 1-Step 不能证明下一轮 Rollout 已使用新权重，而 2-Step 更有闭环验证价值；
- 原始日志、SwanLab、Grafana、资源平台和 Profiler 的观测分工；
- 为什么数据 Loader 成功仍不能证明 Reward 语义正确；
- tmux、容器内临时日志与持久化存储的生命周期边界；
- MoE 为什么是逐 Token、逐 Layer 路由，而不是按整个 Prompt 固定选择一个 Expert；
- Dense/MoE 与 MHA/GQA/MQA 为什么是不同分类维度；
- TP Shard、EP Local/Global Expert、PP Layer Offset 分别是什么；
- HF Checkpoint、分布式 Checkpoint 和在线推理权重的区别；
- Adapter、Bridge/mbridge 与 Converter 分别负责什么；
- 为什么完整 MTP 支持至少包含构建、Loss、保存、同步和推理五个契约；
- Load-only、Train-only、Rollout-only 与 Full MTP 的区别；
- 为什么 MTP-off 向后兼容不只是“不报错”。
- Residual 为什么是“原表示 + 子层修正”，它怎样帮助深层训练；
- MLP/FFN、Gated MLP Gate 与 MoE Router 分别是什么；
- Router 选中 Top-K 后，为什么 K 个 Expert 都参与并加权聚合；
- 为什么 MoE Layer 输出 Hidden State，而不是自然语言 Token；
- Qwen3 系列、Qwen3 MoE 架构子家族和 Qwen3-30B-A3B 具体规格怎样区分；
- MTP Head 为什么不一定只是 Linear Head；
- Checkpoint 能力事实与 YAML 运行意图为什么必须分开；
- MTP Off、Load Existing、Init New 与 Train/Rollout 怎样组合；
- 为什么新增 MTP Module 必须在 DDP/Optimizer 创建之前完成；
- 为什么不能用全模型 `strict=False` 掩盖源 Checkpoint 没有 MTP；
- `detach_encoder=true` 为什么只阻断 MTP Loss 到主干的梯度，而不等于冻结主干；
- 为什么 MTP + CP>1 会产生跨 Rank Future Label/Embedding 依赖；
- 为什么 MTP 是 Speculative Decoding 的一种 Proposer，而二者不能画等号。
- 为什么 Ray Head、DP 数据协调角色和 Megatron 训练 Engine 是三个不同层次；
- `[B,S]` 分别表示什么，Packed Sequence 为什么仍需保存 Segment Boundary；
- 为什么 MTP-1/MTP-2 分别对应固定 `t+2`/`t+3`，而不是一个位置预测所有未来 Token；
- 为什么 `mtp_loss_mask` 不只处理 Padding，还要处理 Prompt、Segment 尾部、越界和跨样本 Shift；
- 为什么动态长度 RL Batch 不能简单平均每个 Microbatch 的平均 Loss；
- Loss Reduction/归约是什么意思，numerator/count 为什么要跨 DP 汇总；
- `λ_mtp`、Token/Microbatch Weight 和 Optimizer Loss Scale 的差别；
- 混合精度为什么同时使用 BF16/FP16 与 FP32，Loss Scaling 为什么不改变训练目标；
- 为什么 MTP Block 的 Dense/MoE 结构不能由推理后端临时选择。
- DCP、HF Checkpoint 和 Online Weights 分别服务什么，为什么不能互相替代；
- Effective Config、Manifest、Tensor、Training State 和 Version State 分别解决什么问题；
- 为什么参数转换需要恢复 TP Shape、PP Layer ID 和 EP Expert ID，而不只是修改 Key；
- `expected = converted = received = applied` 四个集合分别是什么；
- 为什么 RPC 成功或版本号变化不能单独证明 MTP 权重已生效；
- Prepare、Pause、Transfer、Validate、Commit、Resume 的顺序为什么不能交换；
- Fail-closed 可见性与 Tensor Rollback 有什么区别；
- Completion Marker、Structure Hash 和 Value Checksum 分别验证什么。
- Logit、Probability、Token Logprob 和 Sequence Logprob 怎样转换；
- 为什么 `ratio = exp(current_logp - old_logp)`，以及两侧必须对齐同一 Token；
- 为什么 Rollout/Trainer 分进程不等于 Off-policy；
- PPO 为什么需要 Critic Baseline，GRPO 如何用 Group-relative Baseline 替代；
- 为什么 LLM On-policy RL 经常受 Rollout 限制，但不能只优化推理 TPS；
- 为什么 Local BS 很关键但不是唯一物理变量；
- 为什么 MTP 在低 Local BS 下更容易加速，高 Local BS 下收益可能下降；
- 为什么 Acceptance Rate 不能单独解释 Speedup；
- 为什么正式 Benchmark 要与 Profiler 分开，并同时报告 Mean 与 Percentile。

## 明确延期但不可遗忘

### PPO / GRPO 专题

当前状态：已补齐 On-policy/Off-policy、Logprob、PPO/GRPO 角色、Ratio/Clip、Critic Baseline 与
Group-relative Baseline 的最小完整骨架。

仍待深入：GAE、Critic Loss、Entropy Bonus、Token-level Advantage、长度偏差和具体框架变体。
继续保持“不阻塞 MTP 主线、遇到实际 Loss 代码时按需回补”的节奏。

最新入口：[2026-08-06 On-policy 训推、Logprob 与 MTP 性能分析](../notes/2026/08/OnPolicy训推与MTP性能分析_20260806.md)。

## 间隔复习队列

| 笔记 | 首次学习 | D+1 | D+3 | D+7 | D+14 | D+30 |
|---|---|---|---|---|---|---|
| [LLM 普通模型 Baseline 前的基础理论](../notes/2026/07/基础理论_20260723.md) | 2026-07-23 | 2026-07-24 | 2026-07-26 | 2026-07-30 | 2026-08-06 | 2026-08-22 |
| [从预训练到 RL 后训练：MTP 与状态管理基础](../notes/2026/07/训练与RL后训练基础_20260724.md) | 2026-07-24 | ✅ 2026-07-26 | 2026-07-27 | 2026-07-31 | 2026-08-07 | 2026-08-23 |
| [RLVR 策略角色、Logprob 时序与训练诊断](../notes/2026/07/RLVR策略版本与训练诊断_20260726.md) | 2026-07-26 | 2026-07-27 | 2026-07-29 | 2026-08-02 | 2026-08-09 | 2026-08-25 |
| [MTP Label、Loss 与参数更新](../notes/2026/07/MTP标签损失与参数更新_20260726.md) | 2026-07-26 | 2026-07-27 | 2026-07-29 | 2026-08-02 | 2026-08-09 | 2026-08-25 |
| [MTP 推测解码：自回归串行、成块 Verify 与状态提交](../notes/2026/07/MTP推测解码与成块验证_20260726.md) | 2026-07-26 | 2026-07-27 | 2026-07-29 | 2026-08-02 | 2026-08-09 | 2026-08-25 |
| [训练部署 Baseline 与 RLVR 单步闭环](../notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md) | 2026-07-27 | ✅ 2026-07-28 | 2026-07-30 | 2026-08-03 | 2026-08-10 | 2026-08-26 |
| [Qwen3 MoE 与 MTP 适配最小架构](../notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md) | 2026-07-27 | ✅ 2026-07-28 | 2026-07-30 | 2026-08-03 | 2026-08-10 | 2026-08-26 |
| [Transformer 残差、MLP 与 MoE 路由](../notes/2026/07/Transformer残差MLP与MoE路由_20260728.md) | 2026-07-28 | 2026-07-29 | 2026-07-31 | 2026-08-04 | 2026-08-11 | 2026-08-27 |
| [MTP Head 状态机与训练适配边界](../notes/2026/07/MTPHead状态机与训练适配边界_20260728.md) | 2026-07-28 | 2026-07-29 | 2026-07-31 | 2026-08-04 | 2026-08-11 | 2026-08-27 |
| [从 RL Trajectory 到 Megatron：Packed Sequence、MTP Mask 与 Loss 归一化](../notes/2026/07/RL轨迹到Megatron与MTP损失归一化_20260728.md) | 2026-07-28 | 2026-07-29 | 2026-07-31 | 2026-08-04 | 2026-08-11 | 2026-08-27 |
| [MTP 模型状态流与在线权重事务](../notes/2026/07/MTP模型状态流与在线权重事务_20260729.md) | 2026-07-29 | 2026-07-30 | 2026-08-01 | 2026-08-05 | 2026-08-12 | 2026-08-28 |
| [On-policy 训推、Logprob 与 MTP 性能分析](../notes/2026/08/OnPolicy训推与MTP性能分析_20260806.md) | 2026-08-06 | 2026-08-07 | 2026-08-09 | 2026-08-13 | 2026-08-20 | 2026-09-05 |
| [推理算子位置图与性能归因](../notes/2026/08/推理算子位置图与性能归因_20260811.md) | 2026-08-11 | 2026-08-12 | 2026-08-14 | 2026-08-18 | 2026-08-25 | 2026-09-10 |
| [MTP 与 Context Parallel 训练正确性](../notes/2026/08/MTP与ContextParallel训练正确性_20260817.md) | 2026-08-17 | 2026-08-18 | 2026-08-20 | 2026-08-24 | 2026-08-31 | 2026-09-16 |
| [MoE Router Replay 与训推路由一致性](../notes/2026/08/MoERouterReplay与训推路由一致性_20260817.md) | 2026-08-17 | 2026-08-18 | 2026-08-20 | 2026-08-24 | 2026-08-31 | 2026-09-16 |
| [MTP 与 Pipeline Parallel 阶段所有权](../notes/2026/08/MTP与PipelineParallel阶段所有权_20260818.md) | 2026-08-18 | 2026-08-19 | 2026-08-21 | 2026-08-25 | 2026-09-01 | 2026-09-17 |
| [Online RL 的 Cohort 并发与会话容量](../notes/2026/08/OnlineRL的Cohort并发与会话容量_20260818.md) | 2026-08-18 | 2026-08-19 | 2026-08-21 | 2026-08-25 | 2026-09-01 | 2026-09-17 |
| [Online MTP 训练成本、K/D 解耦与性能归因](../notes/2026/08/OnlineMTP训练成本与性能归因_20260826.md) | 2026-08-26 | 2026-08-27 | 2026-08-29 | 2026-09-02 | 2026-09-09 | 2026-09-25 |

完成复习后，可以把日期改成：

```text
✅ 2026-07-24
```

如果某个问题答错：

1. 不要只重复阅读答案；
2. 用自己的话重新解释；
3. 把错误理解和正确理解同时写进当天 Inbox；
4. 在正式笔记中增加“常见误解”；
5. 将该知识点安排到第二天再次复习。
