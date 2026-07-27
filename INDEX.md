# 知识库总索引

> 最后更新：2026-07-27

## 按日期

### 2026 年 7 月

- [2026-07-23：LLM 普通模型 Baseline 前的基础理论](notes/2026/07/基础理论_20260723.md)
- [2026-07-24：从预训练到 RL 后训练——MTP 与状态管理基础](notes/2026/07/训练与RL后训练基础_20260724.md)
- [2026-07-26：RLVR 策略角色、Logprob 时序与训练诊断](notes/2026/07/RLVR策略版本与训练诊断_20260726.md)
- [2026-07-26：MTP Label、Loss 与参数更新](notes/2026/07/MTP标签损失与参数更新_20260726.md)
- [2026-07-26：MTP 推测解码——自回归串行、成块 Verify 与状态提交](notes/2026/07/MTP推测解码与成块验证_20260726.md)

## 按主题

### Baseline 与推理工程

- [Dense 模型、MoE 与 Baseline](notes/2026/07/基础理论_20260723.md#第一部分dense-模型与-baseline)
- [第一条 Baseline 的实践清单](notes/2026/07/基础理论_20260723.md#第十一部分第一条-baseline-的实践清单)
- [推理性能指标](notes/2026/07/基础理论_20260723.md#第十部分推理性能指标)

### Transformer 基础

- [Token、Tokenizer 与自回归生成](notes/2026/07/基础理论_20260723.md#第二部分tokentokenizer-与自回归生成)
- [Transformer、Hidden State 与 FFN](notes/2026/07/基础理论_20260723.md#第三部分transformerhidden-state-与-ffn)
- [Attention、Q/K/V 与各种 Head](notes/2026/07/基础理论_20260723.md#第四部分attentionqkv-与各种-head)

### 推理流程

- [Prefill、Decode 与完整 Token 流程](notes/2026/07/基础理论_20260723.md#第六部分prefilldecode-与完整-token-流程)
- [KV Cache](notes/2026/07/基础理论_20260723.md#第七部分kv-cache)
- [FlashAttention 与 PagedAttention](notes/2026/07/基础理论_20260723.md#第八部分flashattention-与-pagedattention)
- [上下文越长为什么越慢](notes/2026/07/基础理论_20260723.md#第九部分为什么上下文越长越慢)
- [MTP、Draft、Verify 与 Cache 回滚](notes/2026/07/训练与RL后训练基础_20260724.md#7-mtp训练时多步预测推理时多-token-草拟)
- [MTP 训练侧与推理侧怎样接起来](notes/2026/07/MTP标签损失与参数更新_20260726.md#7-训练侧-mtp-与推理侧-mtp-怎样接起来)
- [推测解码：自回归串行与成块 Verify](notes/2026/07/MTP推测解码与成块验证_20260726.md#3-为什么自回归仍能成块-verify)
- [最长接受前缀与 Target 状态提交](notes/2026/07/MTP推测解码与成块验证_20260726.md#5-为什么只能接受最长有效前缀)
- [递归式 MTP Draft 与实际成本](notes/2026/07/MTP推测解码与成块验证_20260726.md#81-一种递归式原生-mtp-draft)
- [KV Cache、递归状态与混合 Attention](notes/2026/07/训练与RL后训练基础_20260724.md#8-从-kv-cache-扩展到通用模型状态)

### 训练与后训练

- [架构、参数、激活值、Loss 与 Checkpoint](notes/2026/07/训练与RL后训练基础_20260724.md#2-训练问题的共同语言)
- [预训练](notes/2026/07/训练与RL后训练基础_20260724.md#3-预训练先学会预测文本中的下一个-token)
- [SFT](notes/2026/07/训练与RL后训练基础_20260724.md#4-sft仍然预测-token但数据变成了理想行为示范)
- [RL 后训练闭环](notes/2026/07/训练与RL后训练基础_20260724.md#5-rl-后训练模型从评价结果中继续学习)
- [Old、Prox、Current、Reference Logprob](notes/2026/07/训练与RL后训练基础_20260724.md#59-相同-token-的四类-logprob到底指什么)
- [Advantage、Policy Ratio 与 Policy Loss](notes/2026/07/训练与RL后训练基础_20260724.md#512-advantage-怎样变成-policy-loss)
- [PPO Clip 与 KL 约束](notes/2026/07/训练与RL后训练基础_20260724.md#513-ppo-clip-与-kl-约束不是一回事)
- [Actor、Rollout 与权重同步](notes/2026/07/训练与RL后训练基础_20260724.md#6-为什么-actor-与-rollout-engine-要分开)
- [从启动命令到一个 RL Step](notes/2026/07/训练与RL后训练基础_20260724.md#65-从启动命令到一个-rl-step-的通用工程链路)
- [模型家族、Checkpoint 与 Actor/Reference/Rollout](notes/2026/07/RLVR策略版本与训练诊断_20260726.md#1-模型家族-checkpoint-与运行角色)
- [W0/W5/W6 策略版本与 Logprob 时序](notes/2026/07/RLVR策略版本与训练诊断_20260726.md#2-w0-w5-w6-策略版本与-logprob-时序)
- [如何判断 RLVR 变好还是变差](notes/2026/07/RLVR策略版本与训练诊断_20260726.md#4-如何判断-rlvr-变好还是变差)
- [NTP 与 MTP Label Shift](notes/2026/07/MTP标签损失与参数更新_20260726.md#2-未来-token-怎样变成-ntp-与-mtp-label)
- [Cross-Entropy 与正确 Token 概率](notes/2026/07/MTP标签损失与参数更新_20260726.md#3-cross-entropy-到底衡量什么)
- [Forward、Loss、Backward 与 Optimizer](notes/2026/07/MTP标签损失与参数更新_20260726.md#4-forward-loss-backward-optimizer-的严格分工)
- [MTP 参数的完整生命周期](notes/2026/07/MTP标签损失与参数更新_20260726.md#6-mtp-参数的完整生命周期)

### 并行与状态管理

- [常见并行策略：DP、TP、PP、EP、CP](notes/2026/07/训练与RL后训练基础_20260724.md#9-常见并行策略的最低认知)
- [训练与推理 Checkpoint](notes/2026/07/训练与RL后训练基础_20260724.md#26-checkpoint)
- [训练和推理如何接起来](notes/2026/07/训练与RL后训练基础_20260724.md#10-与-2026-07-23-笔记的联系)
- [Draft/Target Cache 与状态边界](notes/2026/07/MTP推测解码与成块验证_20260726.md#6-正式状态临时状态与-cache-所有权)

### 快速复习

- [常见概念纠正](notes/2026/07/基础理论_20260723.md#第十二部分当前最容易混淆的概念纠正)
- [一分钟复习卡](notes/2026/07/基础理论_20260723.md#第十四部分一分钟复习卡)
- [训练、RL、MTP 一分钟复习](notes/2026/07/训练与RL后训练基础_20260724.md#13-一分钟复习)
- [训练、RL、MTP 自测问题](notes/2026/07/训练与RL后训练基础_20260724.md#14-自测问题)
- [RLVR 策略版本与训练诊断一分钟复习](notes/2026/07/RLVR策略版本与训练诊断_20260726.md#6-一分钟复习)
- [RLVR 策略版本与训练诊断自测](notes/2026/07/RLVR策略版本与训练诊断_20260726.md#7-自测问题)
- [MTP Label、Loss 与参数更新一分钟复习](notes/2026/07/MTP标签损失与参数更新_20260726.md#9-一分钟复习)
- [MTP Label、Loss 与参数更新自测](notes/2026/07/MTP标签损失与参数更新_20260726.md#10-自测问题)
- [MTP 推测解码一分钟复习](notes/2026/07/MTP推测解码与成块验证_20260726.md#12-一分钟复习)
- [MTP 推测解码自测](notes/2026/07/MTP推测解码与成块验证_20260726.md#13-自测问题)
- [后续学习路线](notes/2026/07/基础理论_20260723.md#第十五部分后续学习路线)

## 待深入专题

- MoE、Router、Expert Parallel 与 All-to-All；
- Batch、并发与 Continuous Batching；
- Tensor Parallel、Pipeline Parallel、Context Parallel 的通信与张量变化（入门分类已覆盖）；
- RoPE 与长上下文扩展；
- FlashAttention kernel；
- 量化与 KV Cache 量化；
- Speculative Decoding 的成块 Verify、线性接受前缀与状态提交已有入门；Sampling 分布校正推导、树形候选与框架实现仍待深入；
- [P1 学习债务：PPO、GRPO 的完整数学与实现](inbox/2026-07-27.md#1425明确延期专题ppo-与-grpo)（入门已覆盖；不阻塞当前 MTP 主线，按触发条件回补）；
- GDN、递归状态与混合 Attention 的具体实现。

## 索引维护规则

新增正式笔记时：

1. 在“按日期”下增加一条；
2. 在一个或多个“按主题”分组中增加链接；
3. 如果笔记解决了“待深入专题”，更新对应状态；
4. 同一知识点优先链接到最完整版本，避免多个近似入口；
5. 保持标题短而明确，让手机端也能快速扫描。
