# Qwen3-MoE 外挂 Native MTP 模块结构

> 日期：2026-07-30  
> 状态：已整理；开源结构已确认，Qwen3-MoE 迁移仍待代码与运行验证

## 整理边界

本文区分三层信息：

- **开源事实**：Megatron-Core 的标准 MTP Layer，以及 vLLM Qwen3.5 Native MTP 的实现结构；
- **当前设计选择**：为不自带 MTP 的 Qwen3-MoE 新增一层同类 Native MTP 模块；
- **待验证项**：目标版本能否使用 Qwen3-MoE block spec 构建、训练和在 vLLM 中同构加载。

它不是公开 Qwen3-30B-A3B 原生声明的结构，也不能仅凭设计文档写成 Trail 已经支持。

公开参考：

- [Megatron-Core Multi-Token Prediction](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/multi_token_prediction.py)
- [Megatron-Core GPTModel](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py)
- [vLLM Qwen3.5 MTP](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/models/qwen3_5_mtp.py)

---

## 1. 一句话定位

一层 Native MTP 模块复用主模型在位置 `t` 的最终上下文表示，再读取已经知道的 `x_{t+1}`，用一层远小于完整 Backbone 的附加 Transformer Layer 预测 `x_{t+2}`：

```text
Backbone hidden h_t ── hnorm ─────────┐
                                      ├─ concat [2H]
Next-token embedding E(x_{t+1}) ─ enorm ┘
→ MTP-owned 2H→H Projection
→ 一层由 Qwen3-MoE Adapter 提供的 MTP Transformer Layer
→ Final Norm
→ 共享 LM Head
→ x_{t+2} Draft Logits
```

第一行中的两路输入不是直接做加法。它们先分别归一化，再沿 Hidden 维拼接。

---

## 2. 用 `[A,B,C,D,E]` 理解两个目标

当前位置为 `C` 时：

```text
主模型：h_C → LM Head → 预测 D
MTP-1：h_C + Embedding(D) → MTP Module → 预测 E
```

概率语义是：

```text
主模型：P(D | A,B,C)
MTP-1： P(E | A,B,C,D)
```

训练时会对整段序列并行构造监督：

```text
h_A + Embedding(B) → 预测 C
h_B + Embedding(C) → 预测 D
h_C + Embedding(D) → 预测 E
```

因此“一层 MTP”不是只训练一个 Token 位置，而是对每个合法位置增加一份 `t+2` 监督。轨迹尾部、Padding、Prompt 区域和 Packed Segment 边界通过 `mtp_loss_mask` 排除。

训练时的 `x_{t+1}` 来自真实 trajectory，属于 Teacher Forcing；它不是先从随机或尚未训练好的 MTP 模块采样出来再作为监督。

---

## 3. 每一步的输入、Shape 和作用

### 3.1 `h_t`：Backbone 的上下文表示

```text
input_ids
→ Embedding
→ Qwen3-MoE 全部 Backbone Layers
→ hidden_states [B,S,H]
```

`h_t` 已经编码位置 `t` 能看到的上下文。MTP 复用这份表示，不重新运行整个 Backbone。

### 3.2 `E(x_{t+1})`：已知下一 Token 的语义

为了预测 `x_{t+2}`，MTP 还需要知道紧邻的 `x_{t+1}`。训练侧把每条 trajectory 的 Token 序列按边界移动一位，再经过共享 Token Embedding 得到：

```text
next_token_embedding [B,S,H]
```

它回答的是：“已经知道下一 Token 是什么后，再往前一步应该是什么？”

### 3.3 两路独立 Norm

`h_t` 已经过多层网络，而 Token Embedding 是参数表的直接输出，两者数值分布不同。分别执行 `hnorm` 和 `enorm`，可以在融合前稳定两路尺度。

### 3.4 Concat 与 `2H→H` Projection

```text
concat(hnorm(h_t), enorm(E(x_{t+1}))) → [B,S,2H]
eh_proj                               → [B,S,H]
```

Concat 保留“上下文状态”和“下一 Token”两种来源；Projection 一方面恢复 Decoder Layer 需要的 `H` 维，另一方面学习两路信息应如何融合。

### 3.5 一层 Qwen3-MoE Transformer Layer

当前 Qwen3-MoE 设计选择是让 Adapter 提供与该模型家族兼容的 Layer Spec：

```text
Fusion Hidden State
→ Causal Self-Attention
→ Residual / Norm
→ Router
→ Top-K MoE Experts
→ Residual
→ MTP Hidden State
```

所以工程语境中的 “MTP Head” 不只是词表 Linear，而是一组附加参数，可能包含：

- 两路 Norm；
- `2H→H` Fusion Projection；
- Attention；
- Router 与 Experts；
- Final Norm。

如果 MTP inner layer 使用 MoE，它也会产生 EP Expert 分片和全局 Expert 编号恢复问题。训练侧与推理侧不能分别选择 Dense/MoE；结构由模型契约确定，必须同构。

### 3.6 Final Norm 与共享 LM Head

MTP hidden 经过 Final Norm，再复用主模型的 `H→V` LM Head 得到：

```text
mtp_logits [B,S,V]
```

共享 LM Head 可以复用相同词表语义并避免复制一个大型词表矩阵。训练时直接用 `x_{t+2}` 做 Cross-Entropy Target。

---

## 4. 哪些参数新增，哪些参数共享

### MTP 专属参数

```text
hnorm / enorm
2H→H eh_proj
MTP Attention
MTP Router / Experts
MTP Final Norm
```

### 与主模型共享

```text
Token Embedding
LM Head
```

源 checkpoint 无 MTP 时，需要受控初始化 MTP 专属参数；Backbone 参数仍应严格加载。共享参数不是需要额外随机初始化的一份副本。

默认隔离 MTP auxiliary gradient 时：

```text
Policy Loss → 更新 Backbone 和原有共享参数
MTP Loss    → 更新 MTP 专属参数
```

MTP Forward 可以读取 Backbone hidden、Embedding 和 LM Head，但通过 detached view 阻止 MTP Loss 沿这些共享路径修改主干。`detach` 只控制 MTP Loss 的梯度边界，不代表整个 Backbone 在 RL 中被冻结。

---

## 5. 一层与多层 MTP

目标深度为 1：

```text
主模型 → 预测 t+1
MTP-1  → 预测 t+2
```

目标深度为 2 时，通常再顺序增加一层 MTP Module：

```text
MTP-1 hidden + E(x_{t+2})
→ MTP-2
→ 预测 x_{t+3}
```

这不是一个 MTP Layer 同时独立输出任意多个未来 Token。深度增加会同步增加参数、Label Shift、Mask、推理状态和权重转换复杂度，所以第一版先固定深度 1。

---

## 6. 推理时怎样变成 Drafter

训练得到有效参数后，vLLM 必须构建同构 MTP 模块：

```text
相同 Norm / Projection / Qwen3-MoE Layer / Final Norm
相同 Embedding 与 LM Head 共享关系
相同参数名、Shape 和 dtype
```

推理概念链为：

```text
Target 得到 h_t，并生成或确认 x_{t+1}
→ MTP 根据 h_t 与 x_{t+1} 提议 x_{t+2}
→ Target Verify
→ Accept：提交连续有效候选
→ Reject：丢弃未接受候选状态并使用 Target 结果
```

MTP 输出的是 Draft，不取代 Target 的最终概率分布。预测不准主要影响接受率和性能，不应改变 Target 的正确分布。

---

## 7. 通用结构与 Qwen3-MoE 特化边界

可复用的通用范式：

```text
Backbone Hidden + Next-token Embedding
→ Fusion
→ Lightweight Prediction Block
→ Future-token Logits
```

Qwen3-MoE Adapter 需要固定的模型特化内容：

- Norm 类型；
- Qwen3-MoE Attention 与位置编码；
- Router、Expert 和 Top-K 结构；
- MTP Layer Spec；
- TP/PP/EP ownership；
- 参数名称、Shape、共享关系和转换规则。

所以公共 Trainer 负责 Label、Mask 和 Loss，Adapter/Bridge 负责模型结构，vLLM 模型层负责同构推理加载。不能把 Qwen3-MoE 参数名硬编码进公共 PPO Trainer。

---

## 8. 开发前必须得到的最小证据

设计合理不等于目标版本已经支持。实现按下面顺序验证：

1. `GPTModel` 接收的 `mtp_block_spec` 确实实例化了一层 MTP Module；
2. Module Tree、参数 Key、Shape 和共享关系符合契约；
3. MTP-off 时这些专属参数消失；
4. 无 MTP checkpoint 只缺失精确的新增参数集合；
5. 一条固定序列的 `t+2` Label/Mask 正确；
6. MTP Loss 有限，MTP 参数 Gradient 非零且 Optimizer 后发生变化；
7. MCore→HF/vLLM 转换后的 Key/Shape 完整一致；
8. vLLM 使用同一版本参数产生 Draft，并输出 Acceptance 指标。

其中 1～3 是 Build-only Spike，不需要先占用完整 RL 训练资源。

---

## 9. 一分钟复习

```text
MTP-1 在位置 t 读取：
主模型 h_t + 真实/已确认的 x_{t+1} Embedding

两路分别 Norm → Concat 成 2H → Projection 回 H
→ 一层模型特定 Transformer Layer
→ Final Norm → 共享 LM Head
→ 预测 x_{t+2}

MTP 专属参数由 MTP Loss 训练；
默认 detach 时，MTP Loss 不修改 Backbone；
训练与 vLLM 推理必须使用完全同构的结构和参数。
```

## 10. 自测问题

1. `h_t + E(x_{t+1})` 中的 `+` 为什么不能理解成直接相加？
2. 为什么预测 `x_{t+2}` 时还需要已知的 `x_{t+1}`？
3. `2H→H Projection` 同时解决哪两个问题？
4. 为什么“一层 MTP Head”不等于一个 Linear？
5. 默认 detach 时，Policy Loss 和 MTP Loss 分别更新哪些参数？
6. 为什么训练侧使用 MoE MTP Layer 后，推理侧不能换成 Dense MTP Layer？
7. 哪些证据只能证明模块构建成功，哪些证据才能证明端到端可用？
