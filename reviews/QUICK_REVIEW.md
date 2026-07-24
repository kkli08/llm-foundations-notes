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

## 间隔复习队列

| 笔记 | 首次学习 | D+1 | D+3 | D+7 | D+14 | D+30 |
|---|---|---|---|---|---|---|
| [LLM 普通模型 Baseline 前的基础理论](../notes/2026/07/基础理论_20260723.md) | 2026-07-23 | 2026-07-24 | 2026-07-26 | 2026-07-30 | 2026-08-06 | 2026-08-22 |

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
