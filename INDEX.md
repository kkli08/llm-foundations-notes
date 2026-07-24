# 知识库总索引

> 最后更新：2026-07-24

## 按日期

### 2026 年 7 月

- [2026-07-23：LLM 普通模型 Baseline 前的基础理论](notes/2026/07/基础理论_20260723.md)

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

### 快速复习

- [常见概念纠正](notes/2026/07/基础理论_20260723.md#第十二部分当前最容易混淆的概念纠正)
- [一分钟复习卡](notes/2026/07/基础理论_20260723.md#第十四部分一分钟复习卡)
- [后续学习路线](notes/2026/07/基础理论_20260723.md#第十五部分后续学习路线)

## 待深入专题

- MoE、Router、Expert Parallel 与 All-to-All；
- Batch、并发与 Continuous Batching；
- Tensor Parallel、Pipeline Parallel、Context Parallel；
- RoPE 与长上下文扩展；
- FlashAttention kernel；
- 量化与 KV Cache 量化；
- Speculative Decoding。

## 索引维护规则

新增正式笔记时：

1. 在“按日期”下增加一条；
2. 在一个或多个“按主题”分组中增加链接；
3. 如果笔记解决了“待深入专题”，更新对应状态；
4. 同一知识点优先链接到最完整版本，避免多个近似入口；
5. 保持标题短而明确，让手机端也能快速扫描。
