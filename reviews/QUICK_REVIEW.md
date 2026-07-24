# 快速复习

> 最后更新：2026-07-24

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
Old Logprob      → 生成数据时的策略概率
Current Logprob  → 当前 Actor 的训练概率
Reference Logprob → KL 约束的参考概率
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
- Old、Current、Reference Logprob 为什么必须对应同一条 Response token；
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
| [从预训练到 RL 后训练：MTP 与状态管理基础](../notes/2026/07/训练与RL后训练基础_20260724.md) | 2026-07-24 | 2026-07-25 | 2026-07-27 | 2026-07-31 | 2026-08-07 | 2026-08-23 |

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
