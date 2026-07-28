# Transformer 残差、MLP 与 MoE 路由

> 日期：2026-07-28
> 来源：当前基础理论对话与工作 Session `019f8e1a-1ff1-7002-aa46-85ffd51a11cc`
> 状态：已整理

## 整理记录

- 这是对 [Qwen3 MoE 与 MTP 适配最小架构](Qwen3MoE与MTP适配最小架构_20260727.md) 的增量深化，不重复昨天已经整理的 TP、EP、PP、Checkpoint 与五个 MTP 契约。
- 今天重点解决：Residual 是什么、MLP/FFN 是什么、Hidden State 怎样逐层变化、Router 怎样选择 Top-K Expert，以及“一层输出”和“生成一个 Token”为什么不是同一件事。
- 模型配置数据只采用公开资料；内部路径、任务状态和未验证的项目实现结论不进入本文。

---

## 0. 先把整条链串起来

一个 Decoder-only Transformer 的单层可以先记成：

```text
输入 Hidden State x
→ Norm
→ Attention：跨 Token 汇聚上下文
→ Residual Add：把 Attention 的修正加回 x
→ Norm
→ Dense MLP，或 Router + Top-K Expert MLP
→ Residual Add：把 MLP/MoE 的修正再加回
→ 当前层输出 Hidden State
→ 送给下一层
```

所有 Transformer Layer 都处理完以后：

```text
最后一层 Hidden State
→ Final Norm
→ LM Head
→ Vocabulary Logits
```

之后才根据运行场景分流：

```text
推理：Logits → Sampling/Argmax → 一个新 Token
监督训练：Logits + Label → Cross-Entropy Loss
RL 训练：Logits → Response Logprob → Policy Loss/KL 等
```

最重要的一句话：

> Transformer 的每一层都在更新 Token 的向量表示；不是每一层都生成一个 Token。一个新 Token 通常要等所有层和 LM Head 都算完以后才产生。

---

## 1. Hidden State 到底是什么

### 1.1 Token ID 不是模型内部的语义表示

Tokenizer 先把文本转换为离散 Token ID：

```text
“今天” → Token ID 18342
```

Token ID 只是词表中的编号，编号大小没有语义距离。模型先通过 Embedding Table 把它变成一个浮点向量：

```text
Token ID
→ Embedding Vector
→ Transformer Layer 0 的输入 Hidden State
```

### 1.2 Hidden State 是“当前层对这个位置的理解”

对序列中的每个位置，模型都维护一个长度为 `hidden_size` 的向量。这个向量经过每层 Attention 和 MLP 后不断变化：

```text
h_i^(0)：初始 Token/Position 表示
h_i^(1)：融合一层上下文后的表示
h_i^(2)：再经过一层特征加工后的表示
...
h_i^(L)：最后一层表示
```

它不是自然语言，也不是词表概率，而是后续计算使用的连续数值表示。

### 1.3 同一 Token 在不同上下文和不同层的 Hidden State 不同

例如“苹果”出现在水果语境和公司语境中，初始 Token Embedding 可以相同，但经过 Attention 汇聚不同上下文后，Hidden State 会不同。

这也解释了 Router 为什么虽然“逐 Token”决策，却仍会受到整个 Prompt 的间接影响：

```text
Prompt 上下文
→ 影响当前 Token 的 Hidden State
→ Hidden State 输入 Router
→ 影响该层选择哪些 Expert
```

---

## 2. Residual 残差连接是什么

### 2.1 最小公式

Residual 的核心形式是：

```text
output = input + sublayer(input)
```

在常见 Pre-Norm Transformer 中可以简化为：

```text
y = x + Attention(Norm(x))
z = y + MLP(Norm(y))
```

其中：

- `x` 是进入 Attention 子层前的表示；
- `Attention(Norm(x))` 是 Attention 学到的修正量；
- `y` 保留原表示，同时加入 Attention 的新信息；
- MLP/MoE 后再进行一次同样的残差相加。

### 2.2 为什么不直接使用子层输出

如果只写：

```text
output = sublayer(input)
```

每层都必须重新完整表达所有有用信息。残差允许子层把任务变成：

> 在已有表示上学习一个增量修正，而不是从头重写整个表示。

直觉上类似：

```text
原稿 + 本轮修改意见 = 新版本
```

而不是每轮都丢弃原稿重新写。

### 2.3 Residual 的两个主要作用

第一，保存信息通路：如果某个子层暂时没有学到有用变换，它可以输出接近 0，原表示仍能继续传递。

第二，改善梯度传播：反向传播时，梯度除了经过复杂子层，还存在一条加法形成的直接路径，使很深的网络更容易训练。

### 2.4 Residual 不等于 KV Cache

两者作用完全不同：

| 概念 | 作用 |
|---|---|
| Residual | 同一次 Forward 内，把子层输入直接加到子层输出上 |
| KV Cache | 推理 Decode 时，跨生成步保存历史 Token 在各层的 K/V |

Residual 是模型架构的一部分，训练和推理都会使用；KV Cache 主要是自回归推理优化。

---

## 3. MLP 与 FFN 是什么

### 3.1 两个名字为什么经常混用

MLP 是 Multi-Layer Perceptron，多层感知机；FFN 是 Feed-Forward Network，前馈网络。

在 Transformer 语境中，两者通常指每个 Layer 里 Attention 后面的同一类子模块，因此工程文档经常混用：

```text
Transformer MLP ≈ Transformer FFN
```

严格说，MLP 是更广的神经网络类别；FFN 强调信息单向前馈。但在当前任务里，把它们当作近似同义词足够。

### 3.2 普通 FFN 怎样计算

经典结构可以写成：

```text
FFN(x) = W_down · activation(W_up · x)
```

数据流是：

```text
hidden_size
→ Up Projection 扩展到 intermediate_size
→ 非线性激活
→ Down Projection 压回 hidden_size
```

它不是在序列位置之间做查询，而是在每个 Token 自己的 Hidden Vector 内进行特征变换。

### 3.3 Gated MLP / SwiGLU 是什么

现代 LLM 常用门控结构，例如：

```text
GatedMLP(x)
= W_down(SiLU(W_gate x) ⊙ (W_up x))
```

可以直观理解为两条支路：

```text
Gate 支路：决定哪些特征应该通过、通过多少
Up 支路：提供要被处理的内容特征
两者逐元素相乘
→ Down Projection 回 hidden_size
```

这里的 Gate 不是 MoE Router：

| Gate | Router |
|---|---|
| 在一个 MLP 内按特征维度调制激活 | 在多个 Expert MLP 之间为 Token 选路 |
| 通常所有 Token 都执行 | 通常只执行 Top-K Expert |

### 3.4 Attention 与 MLP 的分工

可以先用这句话记：

```text
Attention：让不同 Token 位置交换信息
MLP/FFN：加工每个 Token 当前已经拥有的表示
```

Attention 回答“当前 Token 应从上下文哪些位置读取什么”；MLP 回答“拿到上下文信息后，这个 Token 的内部特征应该怎样重新组合”。

二者缺一不可，也不是上下两套独立模型，而是每个 Transformer Layer 内的两个子层。

---

## 4. Dense FFN 与 MoE FFN 的关系

### 4.1 Dense Layer

Dense Transformer Layer 中，每个 Token 都经过该层同一套 MLP 参数：

```text
Token A Hidden → Shared MLP
Token B Hidden → Shared MLP
Token C Hidden → Shared MLP
```

### 4.2 MoE Layer

常见 MoE Transformer 保留 Attention，用多个 Expert MLP 和 Router 替换 Dense MLP：

```text
Transformer Layer
├── Attention
└── MoE 子层
    ├── Router
    ├── Expert 0：一个独立 MLP/FFN
    ├── Expert 1：一个独立 MLP/FFN
    └── ...
```

因此：

> Expert 通常就是一套独立的 Gated MLP/FFN；MoE 是在“这一层该使用哪几套 FFN 参数”上引入稀疏选择。

Dense/MoE 与 MHA/GQA/MQA 不是同一个分类维度：前者主要描述 FFN，后者描述 Attention Head 的组织。

---

## 5. Router 怎样选择 Top-K Expert

### 5.1 Router 的输入不是原始文字

对第 `l` 层、第 `i` 个 Token，Router 接收当前上下文化 Hidden State `h_i^(l)`：

```text
h_i^(l)
→ 线性投影 W_router h_i^(l)
→ 每个 Expert 一个分数
```

假设有 4 个 Expert：

```text
Router scores = [2.1, -0.4, 1.5, 0.3]
```

经过 Softmax 得到路由权重，例如：

```text
Router probabilities = [0.56, 0.05, 0.31, 0.08]
```

若 Top-K=2，则选择 Expert 0 和 Expert 2。

### 5.2 被选中的 Top-K 是否都要计算

是，从逻辑计算上看，被选中的 Top-K Expert 都会对这个 Token 的 Hidden State 执行各自的 FFN：

```text
e0 = Expert0(h)
e2 = Expert2(h)
```

再按 Router 权重加权求和：

```text
moe_output = w0 · e0 + w2 · e2
```

有些实现会把选中权重重新归一化，使 `w0 + w2 = 1`；具体数值细节取决于实现和配置。

然后仍然要进行残差相加：

```text
layer_output = previous_hidden + moe_output
```

### 5.3 没选中的 Expert 呢

对于当前 Token，没选中的 Expert 不执行主要 FFN 计算，因此每 Token 激活参数少于模型总参数。

但需要注意：

- 所有 Expert 权重仍需存储；
- 同一批中的其他 Token 可能选择不同 Expert；
- 分布式系统可能需要把 Token 发到 Expert 所在设备；
- Router/负载均衡与通信会产生额外开销。

因此“激活参数少”不等于“推理一定更快”。

### 5.4 Router 选的是 Token 当前表示，不是固定语义标签

不要把 Expert 过度人格化为固定的“数学专家”或“代码专家”。更准确的是：

```text
每层、每个 Token 的 Hidden State
→ Router 计算分数
→ 选择当下最合适的 Top-K 参数路径
```

不同 Token、不同 Layer 都可选择不同 Expert。训练后可能观察到一定功能分化，但这不是一个人工固定分类器。

### 5.5 Load Balance 为什么以后还要学

如果大量 Token 总被路由到少数 Expert：

- 热门 Expert 可能过载；
- 其他 Expert 训练不足；
- 跨设备通信和等待增大；
- 实际吞吐可能下降。

所以完整 MoE 训练还会涉及负载均衡目标、容量控制和 Token Dispatch。但对当前第一版 MTP 适配，先知道风险和观测点即可，不必推导 Router 梯度或通信 Kernel。

---

## 6. MoE Layer 输出的仍是 Hidden State

假设当前有三个 Token：

```text
[A, B, C]
```

一层 MoE 为每个 Token 选择 Expert 并加工向量后，输出仍是同样三个序列位置的向量：

```text
[h_A^(l+1), h_B^(l+1), h_C^(l+1)]
```

序列长度通常没有因为 MoE 而变成新 Token 序列，也没有在每个 Expert 中生成自然语言。

这些 Hidden States 会继续进入下一层。只有最后一层完成后，LM Head 才把目标位置的 Hidden State 映射到词表 Logits。

---

## 7. 训练侧与推理侧怎样使用最终 Hidden State

### 7.1 两边都会算 Logits

训练和推理都使用同一套 Transformer 架构完成 Forward：

```text
Token IDs
→ Embedding
→ 所有 Transformer Layers
→ Final Hidden State
→ LM Head
→ Logits
```

差别主要发生在 Logits 之后，以及是否反向传播。

### 7.2 推理

```text
最后位置 Logits
→ Temperature / Top-k / Top-p / Sampling
→ 新 Token y1
→ 把 y1 放回上下文
→ 再次 Decode 得到 y2
→ ...
```

`y1、y2、y3` 是连续 Decode 步产生的多个 Token，不是多个 Transformer Layer 分别产生的 Token。Detokenizer 最后把这些 Token 转回自然语言文本。

### 7.3 监督训练

训练时完整目标序列已知，可以并行得到多个位置的 Logits：

```text
Logits + 正确 Token Label
→ Cross-Entropy
→ Backward
→ Optimizer Step
```

### 7.4 RL 后训练

RL 训练通常从同一批 Response 的 Logits 得到选中 Token 的 Logprob：

```text
Logits
→ Response Token Logprob
→ Ratio / Advantage / KL / Policy Loss
→ Backward
→ Optimizer Step
```

如果加入 MTP，则模型还可能产生额外 MTP Logits 和 Auxiliary Loss；它们是否影响主干由梯度边界策略决定。

---

## 8. Qwen、Qwen3、Qwen3-30B-A3B 的层级关系

可以按下面层级理解：

```text
Qwen
└── Qwen3 系列/一代模型
    ├── Dense 架构分支
    └── MoE 架构分支
        ├── Qwen3-30B-A3B
        └── 其他 Qwen3 MoE 规格
```

- `Qwen`：更大的品牌/家族；
- `Qwen3`：一个代际或系列；
- `Qwen3 MoE`：架构子家族；
- `Qwen3-30B-A3B`：一个具体模型规格；
- Base、Instruct、后训练版本：通常是不同 Checkpoint/权重版本。

因此工程上的 Model Adapter 往往按 `model_type` 或架构子家族复用，而不是为每个参数规模复制一套逻辑。30B 与其他 Qwen3 MoE 模型可能共享构建策略，但 Hidden Size、层数、Expert 数、Top-K、Attention Head 数等仍由各自 Config 决定。

### 8.1 公开 Qwen3-30B-A3B 例子

截至本文记录的公开 Base Config，Qwen3-30B-A3B 包含：

- 48 个 Decoder Layer；
- 128 个 Expert；
- 每个 Token 激活 8 个 Expert；
- GQA：32 个 Query Head、4 个 KV Head；
- 公开 HF 实现按配置在 Decoder Layer 的 MLP 位置构建稀疏 MoE Block。

这说明一台具体模型可以同时是：

```text
FFN 维度：MoE
Attention 维度：GQA
网络骨架：Decoder-only Transformer
```

这些标签不冲突。

---

## 9. 今天最容易混淆的概念

| 容易误解 | 更准确的说法 |
|---|---|
| 每个 Transformer Layer 生成一个 Token | 每层更新 Hidden State；所有层完成后由 LM Head 产生 Logits |
| MoE Expert 生成新 Token | Expert 只生成当前层的新 Hidden 表示 |
| MLP 和 FFN 是完全不同模块 | Transformer 语境中通常近似指同一个前馈子层 |
| Gated MLP 的 Gate 就是 Router | Gate 调制 MLP 内部特征；Router 在多个 Expert 间选路 |
| MoE 替换整个 Transformer | 常见 MoE 主要替换 Dense FFN，Attention 仍保留 |
| Router 按整个 Prompt 选择一个 Expert | Router 通常逐层、逐 Token，根据上下文化 Hidden State 选 Top-K |
| Top-K 只算排名第一的 Expert | 选中的 K 个 Expert 都逻辑参与计算，再加权聚合 |
| 激活参数少，所以一定更快 | 还要考虑存储、路由、Dispatch、通信、负载均衡和硬件利用率 |
| Residual 是一种推理 Cache | Residual 是层内加法信息通路；KV Cache 才是跨 Decode 步缓存 |
| 训练侧只算 Loss，不需要 LM Head | 训练也先通过 LM Head 得到 Logits，再用 Label/策略目标算 Loss |

---

## 10. 一分钟复习

1. Hidden State 是每层对每个 Token 当前语义与上下文的连续向量表示。
2. Attention 负责跨 Token 汇聚信息，MLP/FFN 负责加工每个 Token 自己的特征。
3. Residual 把子层输入直接加回输出，使子层学习增量修正并改善深层梯度传播。
4. Transformer 中 MLP 与 FFN 通常近似同义；现代模型常用 SwiGLU 等 Gated MLP。
5. 常见 MoE 用 Router + 多个 Expert MLP 替换 Dense MLP，不替换 Attention。
6. Router 在每层按每个 Token 的 Hidden State打分，选 Top-K Expert。
7. 选中的 K 个 Expert 都计算，输出按路由权重聚合后再走 Residual。
8. MoE Layer 输出仍是 Hidden State；所有层结束后才由 LM Head 产生词表 Logits。
9. 推理从 Logits 采样 Token；训练从 Logits 计算监督或 RL Loss，再反向更新参数。
10. Qwen3 是系列，Qwen3 MoE 是架构子家族，Qwen3-30B-A3B 是具体规格。

---

## 11. 自测问题

### 问题 1

Residual 为什么可理解为“原表示 + 学到的修正”？它怎样帮助深层训练？

### 问题 2

Attention 与 MLP 分别主要处理“跨 Token 信息”和“Token 内部特征”中的哪一个？

### 问题 3

Gated MLP 的 Gate 与 MoE Router 有什么区别？

### 问题 4

Top-K=2 时，Router 选中两个 Expert 后，最终 MoE 输出怎样得到？

### 问题 5

为什么说 Router 逐 Token 工作，却又会间接受整个 Prompt 影响？

### 问题 6

为什么 MoE Layer 的输出不是一个自然语言 Token？

### 问题 7

推理和训练在 `Hidden State → LM Head → Logits` 这部分相同，之后分别怎样分流？

### 问题 8

Dense/MoE 与 MHA/GQA/MQA 为什么可以同时描述同一个模型？

---

## 12. 当前学习边界

当前必须掌握：

- Pre-Norm Layer、Residual、Attention、MLP 的连接关系；
- Dense MLP 与 Expert MLP 的关系；
- Router 的 Scores、Top-K、加权聚合；
- Hidden State、Logits、Token 的层级；
- 模型系列、架构子家族、具体模型规格的区别。

可以等实际性能或训练问题出现后再深入：

- Router 梯度和 Top-K 的不可微处理；
- Auxiliary Load-Balancing Loss；
- Capacity Factor、Token Drop；
- Expert Parallel All-to-All；
- Grouped GEMM 和 Expert Kernel 优化。

---

## 13. 参考材料

- [Qwen 官方：Qwen3](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3-30B-A3B Base Config（固定版本）](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/5058eb11c39a850fcc2fd0c83a40a705cad9ca7a/config.json)
- [Hugging Face Transformers：Qwen3 MoE 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
- [Daily Dose of Data Science：Transformer vs. Mixture of Experts](https://www.dailydoseofds.com/p/transformer-vs-mixture-of-experts-in-llms/)：适合看 Dense FFN、Router 与 Expert 的直观图；具体速度、Expert 规模和路由实现仍以目标模型配置与官方实现为准。
- [Switch Transformers](https://www.jmlr.org/beta/papers/v23/21-0998.html)：MoE 路由、负载与稀疏激活的经典参考。
