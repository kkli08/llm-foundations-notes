# 快速复习

> 最后更新：2026-07-26

## 开始新学习前

用 5～10 分钟完成：

1. 阅读本页“当前核心心智模型”；
2. 阅读最近一篇正式笔记的“一分钟复习卡”；
3. 不看答案回答最近一篇笔记的自测问题；
4. 答错的内容加入当天 Inbox。

## 当前核心心智模型

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
→ 主模型 Verify
→ 提交接受的连续前缀
→ 回滚未接受 token 的 KV / 其他状态
```

## 当前必会解释

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
- DP、TP、PP、EP、CP 分别切分什么。

## 间隔复习队列

| 笔记 | 首次学习 | D+1 | D+3 | D+7 | D+14 | D+30 |
|---|---|---|---|---|---|---|
| [LLM 普通模型 Baseline 前的基础理论](../notes/2026/07/基础理论_20260723.md) | 2026-07-23 | 2026-07-24 | 2026-07-26 | 2026-07-30 | 2026-08-06 | 2026-08-22 |
| [从预训练到 RL 后训练：MTP 与状态管理基础](../notes/2026/07/训练与RL后训练基础_20260724.md) | 2026-07-24 | ✅ 2026-07-26 | 2026-07-27 | 2026-07-31 | 2026-08-07 | 2026-08-23 |
| [RLVR 策略角色、Logprob 时序与训练诊断](../notes/2026/07/RLVR策略版本与训练诊断_20260726.md) | 2026-07-26 | 2026-07-27 | 2026-07-29 | 2026-08-02 | 2026-08-09 | 2026-08-25 |
| [MTP Label、Loss 与参数更新](../notes/2026/07/MTP标签损失与参数更新_20260726.md) | 2026-07-26 | 2026-07-27 | 2026-07-29 | 2026-08-02 | 2026-08-09 | 2026-08-25 |

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
