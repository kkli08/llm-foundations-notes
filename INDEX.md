# 知识库总索引

> 最后更新：2026-07-29

## 按日期

### 2026 年 7 月

- [2026-07-23：LLM 普通模型 Baseline 前的基础理论](notes/2026/07/基础理论_20260723.md)
- [2026-07-24：从预训练到 RL 后训练——MTP 与状态管理基础](notes/2026/07/训练与RL后训练基础_20260724.md)
- [2026-07-26：RLVR 策略角色、Logprob 时序与训练诊断](notes/2026/07/RLVR策略版本与训练诊断_20260726.md)
- [2026-07-26：MTP Label、Loss 与参数更新](notes/2026/07/MTP标签损失与参数更新_20260726.md)
- [2026-07-26：MTP 推测解码——自回归串行、成块 Verify 与状态提交](notes/2026/07/MTP推测解码与成块验证_20260726.md)
- [2026-07-27：训练部署 Baseline 与 RLVR 单步闭环](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md)
- [2026-07-27：Qwen3 MoE 与 MTP 适配最小架构](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md)
- [2026-07-28：Transformer 残差、MLP 与 MoE 路由](notes/2026/07/Transformer残差MLP与MoE路由_20260728.md)
- [2026-07-28：MTP Head 状态机与训练适配边界](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md)
- [2026-07-28：从 RL Trajectory 到 Megatron——Packed Sequence、MTP Mask 与 Loss 归一化](notes/2026/07/RL轨迹到Megatron与MTP损失归一化_20260728.md)
- [2026-07-29：MTP 模型状态流与在线权重事务](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md)

## 按主题

### Baseline 与推理工程

- [Dense 模型、MoE 与 Baseline](notes/2026/07/基础理论_20260723.md#第一部分dense-模型与-baseline)
- [第一条 Baseline 的实践清单](notes/2026/07/基础理论_20260723.md#第十一部分第一条-baseline-的实践清单)
- [推理性能指标](notes/2026/07/基础理论_20260723.md#第十部分推理性能指标)
- [资源平台、Ray、训练编排层、Megatron 与 vLLM](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#2-先把系统分层每层只回答一种问题)
- [Baseline、Smoke、效果实验与性能实验](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#4-baselinesmoke效果实验和性能实验)
- [Baseline 的分层成功证据](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#8-如何证明一次-baseline-真正跑通)
- [日志、SwanLab、Grafana 与 Profiler 的分工](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#10-观测工具怎样分工)

### Transformer 基础

- [Token、Tokenizer 与自回归生成](notes/2026/07/基础理论_20260723.md#第二部分tokentokenizer-与自回归生成)
- [Transformer、Hidden State 与 FFN](notes/2026/07/基础理论_20260723.md#第三部分transformerhidden-state-与-ffn)
- [Attention、Q/K/V 与各种 Head](notes/2026/07/基础理论_20260723.md#第四部分attentionqkv-与各种-head)
- [Residual：原表示加子层修正](notes/2026/07/Transformer残差MLP与MoE路由_20260728.md#2-residual-残差连接是什么)
- [MLP、FFN 与 Gated MLP](notes/2026/07/Transformer残差MLP与MoE路由_20260728.md#3-mlp-与-ffn-是什么)
- [Hidden State、Logits 与 Token 的层级](notes/2026/07/Transformer残差MLP与MoE路由_20260728.md#7-训练侧与推理侧怎样使用最终-hidden-state)

### MoE 与模型适配

- [Dense FFN 与 MoE FFN](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#2-从-dense-ffn-到-moe-ffn)
- [Router 为什么逐 Token、逐 Layer 路由](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#23-router-是逐-token逐-layer-工作)
- [TP、EP、PP 与 Local/Global 编号](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#3-tpeppp-在拆什么)
- [HF Checkpoint、Megatron 与 vLLM 的参数世界](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#4-为什么同一模型会有三种参数世界)
- [Adapter、Bridge、Converter 与参数 ABI](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#5-adapterbridge-与-converter)
- [Router Top-K、Expert 计算与加权聚合](notes/2026/07/Transformer残差MLP与MoE路由_20260728.md#5-router-怎样选择-top-k-expert)
- [Qwen3 系列、架构子家族与具体模型规格](notes/2026/07/Transformer残差MLP与MoE路由_20260728.md#8-qwenqwen3qwen3-30b-a3b-的层级关系)
- [Model Adapter 的职责与训练边界](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#4-model-adapter-到底负责什么)
- [Adapter、Converter 与 Loader 的状态流职责](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md#5-adapterconverter-和-loader-的职责)

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
- [一个外层 RLVR Step 与多回答 Group](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#6-一个外层-rlvr-step-到底发生了什么)
- [组内 Advantage 与二元 Reward](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#7-同一-prompt-的多回答与组内-advantage)
- [MTP 完整支持的五个契约](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#7-完整-mtp-支持的五个契约)
- [Load、Train、Rollout MTP 模式矩阵](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#8-为什么-mtp-要拆成-loadtrainrollout-三个开关)
- [MTP-off 向后兼容](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#9-怎样保证向后兼容)
- [MTP Head 不一定只是 Linear Head](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#1-mtp-head-不一定只是一个-linear-head)
- [MTP Off、Load、Init 与 Train/Rollout 状态机](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#3-offload-与-init模型构建状态机)
- [MTP Label、Mask 与 RL Total Loss](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#7-每个训练-step-怎样训练-mtp)
- [`detach_encoder` 与梯度边界](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#8-detach_encoder-的梯度边界)
- [MTP 与 Speculative Decoding 的关系](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#11-speculative-decoding-与-mtp-的关系)
- [Trajectory、Packed Sequence 与训练 Tensor](notes/2026/07/RL轨迹到Megatron与MTP损失归一化_20260728.md#3-trajectory-到训练-tensor-的对象变化)
- [Segment-aware MTP Label 与 Mask](notes/2026/07/RL轨迹到Megatron与MTP损失归一化_20260728.md#5-ntpmtp-label-与-segment-aware-mask)
- [动态 Token Loss 归一化与三种 Scale](notes/2026/07/RL轨迹到Megatron与MTP损失归一化_20260728.md#6-loss-reduction归约是什么意思)
- [DCP、HF Export 与 Online Weight Sync](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md#3-dcphf-checkpoint-和-online-weights-的区别)

### 并行与状态管理

- [常见并行策略：DP、TP、PP、EP、CP](notes/2026/07/训练与RL后训练基础_20260724.md#9-常见并行策略的最低认知)
- [训练与推理 Checkpoint](notes/2026/07/训练与RL后训练基础_20260724.md#26-checkpoint)
- [训练和推理如何接起来](notes/2026/07/训练与RL后训练基础_20260724.md#10-与-2026-07-23-笔记的联系)
- [Draft/Target Cache 与状态边界](notes/2026/07/MTP推测解码与成块验证_20260726.md#6-正式状态临时状态与-cache-所有权)
- [tmux、临时日志与持久化边界](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#11-tmux临时文件与持久化存储)
- [MTP 参数分片、全局编号与在线同步](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#74-契约四训练到推理的转换与在线同步)
- [MTP Training 与 CP>1 的跨 Rank 依赖](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#9-mtp-training-为什么容易和-cp1-冲突)
- [DP、PP 与动态 Token Loss 所有权](notes/2026/07/RL轨迹到Megatron与MTP损失归一化_20260728.md#9-dppp-与-loss-所有权的最低理解)
- [模型状态的五类对象](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md#2-一份模型状态不只有-tensor)
- [Exact-set 参数完整性门禁](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md#6-exact-set-gate怎样证明参数真的完整流转)
- [在线权重版本事务与 Fail-closed](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md#7-在线同步的版本事务)
- [Structure Hash 与 Value Checksum](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md#10-structure-hash-与-value-checksum)

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
- [训练部署 Baseline 一分钟复习与自测](notes/2026/07/训练部署Baseline与RLVR闭环_20260727.md#15-一分钟复习)
- [Qwen3 MoE/MTP 适配一分钟复习与自测](notes/2026/07/Qwen3MoE与MTP适配最小架构_20260727.md#14-一分钟复习)
- [Transformer/MoE 深化一分钟复习与自测](notes/2026/07/Transformer残差MLP与MoE路由_20260728.md#10-一分钟复习)
- [MTP 训练适配一分钟复习与自测](notes/2026/07/MTPHead状态机与训练适配边界_20260728.md#16-一分钟复习)
- [Trajectory、MTP Mask 与 Loss 归一化一分钟复习](notes/2026/07/RL轨迹到Megatron与MTP损失归一化_20260728.md#13-一分钟复习)
- [MTP 状态流与在线权重事务一分钟复习](notes/2026/07/MTP模型状态流与在线权重事务_20260729.md#13-一分钟复习)
- [后续学习路线](notes/2026/07/基础理论_20260723.md#第十五部分后续学习路线)

## 待深入专题

- MoE、Router Scores/Top-K/加权聚合、TP/EP/PP 与 Local/Global 编号已有任务所需入门；Router 梯度、Load Balance Loss、All-to-All Kernel 和 EP 性能优化仍待深入；
- Batch、Microbatch、Padding/Packing 和动态 Token Loss 归一化已有最小认知；推理并发与 Continuous Batching 仍待深入；
- Tensor Parallel、Pipeline Parallel、Context Parallel 的通信与张量变化（入门分类已覆盖；MTP+CP 的跨 Rank Future 依赖已有最小认知）；
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
