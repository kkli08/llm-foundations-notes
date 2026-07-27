# Qwen3 MoE 与 MTP 适配最小架构

> 日期：2026-07-27
> 来源：Codex Session `019f8e1a-1ff1-7002-aa46-85ffd51a11cc`
> 状态：已整理

## 整理记录

- 2026-07-27 20:58：围绕当前 Qwen3 30B MTP 适配任务，整理 MoE、TP/EP/PP、Checkpoint、Bridge、Adapter、参数 ABI 和 MTP 训推开关。
- 只保留可跨项目复用的原理；当日代码审计中的项目路径、内部 Checkpoint 和未验证实现细节均未写入。
- 本文只补当前适配最需要的模型与工程知识，不展开 Router 梯度、Load Balance Loss、All-to-All Kernel 或 Grouped GEMM 性能实现。
- 训练运行和单步验收见 [训练部署 Baseline 与 RLVR 单步闭环](训练部署Baseline与RLVR闭环_20260727.md)。

---

## 0. 今天最终要解决的问题

1. MoE 是不是“按 Prompt 类型选择一个专家”？
2. Dense FFN 与 MoE FFN 是什么关系？
3. TP、EP、PP 分别切什么，为什么会产生 Local/Global 编号？
4. HF Checkpoint、分布式 Checkpoint 和在线 vLLM 权重是什么关系？
5. Adapter、Bridge/mbridge 和 Converter 分别负责什么？
6. 为什么 MTP 适配不是打开一个 YAML 开关？
7. Load MTP、Train MTP 和 Rollout MTP 为什么要分离？
8. 怎样保证新增 MTP 后原有 MTP-off baseline 不受影响？

---

## 1. 先给结论

### 1.1 MoE 的最短心智模型

普通 Transformer Layer 可以简化为：

```text
Hidden States
→ Attention：不同 Token 之间交换信息
→ Dense FFN：每个 Token 经过同一套前馈网络
```

MoE Layer 则是：

```text
Hidden States
→ Attention
→ Router 为每个 Token 选择 Top-K Expert
→ Token 只经过被选中的 Expert FFN
→ 按 Router 权重合并 Expert 输出
```

最重要的纠正：

> MoE 通常不是先判断整个 Prompt 属于数学类还是代码类，再让整段 Prompt 固定走某个专家；Router 通常在每个 MoE Layer 中，对每一个 Token 分别路由。

### 1.2 三种并行一句话记忆

```text
TP：拆同一个大矩阵
EP：拆不同 Expert
PP：拆不同 Layer
```

### 1.3 MTP 适配的本质

完整 MTP 支持至少跨越五个契约：

```text
Checkpoint/模型构建
→ Future Label 与 MTP Loss
→ 保存和恢复
→ 训练权重到推理权重的转换与同步
→ 推理侧 Draft/Verify 与指标
```

所以：

> “配置能够解析”只证明开关存在；“模型里能看到 MTP 参数”也只证明模块存在，都不等于端到端 MTP 已经可用。

### 1.4 最合理的设计边界

```text
通用层：开关、Label、Loss、校验、指标、同步协议
模型 Adapter：Checkpoint 字段、模块结构、参数名称和分片映射
推理后端：Speculative Config、Verify、Acceptance 指标
```

任务先落在 Qwen3 30B，不代表实现应该硬编码成只服务一个具体 Checkpoint。

---

## 2. 从 Dense FFN 到 MoE FFN

### 2.1 普通 Dense Transformer Layer

典型 Decoder Layer：

```text
Input Hidden States
→ Norm
→ Self-Attention
→ Residual Add
→ Norm
→ Dense FFN / MLP
→ Residual Add
→ Output Hidden States
```

这里 `Dense` 的含义是：

> 每个 Token 都使用该层同一套 FFN 参数。

这不表示每个神经元的激活一定都非零，也不是在描述 Attention 是否稀疏。

### 2.2 MoE 替换的是哪一部分

常见 MoE Transformer 保留 Attention，把 Dense FFN 替换为多个 Expert FFN：

```text
Transformer Layer
├── Attention
└── MoE FFN
    ├── Router
    ├── Expert 0
    ├── Expert 1
    ├── ...
    └── Expert N-1
```

因此 Dense/MoE 与 MHA/GQA/MQA 是不同分类维度：

- Dense/MoE：主要描述 FFN 部分是否使用稀疏专家；
- MHA/GQA/MQA：描述 Attention 中 Query Head 与 KV Head 的组织方式。

一个 MoE 模型完全可以同时使用 GQA。

### 2.3 Router 是逐 Token、逐 Layer 工作

假设某层有 8 个 Expert，Top-K=2。

同一个 Prompt 的不同 Token 可能被路由为：

```text
Token A → Expert 1、6
Token B → Expert 0、3
Token C → Expert 1、5
```

同一个 Token 到下一层时，又可能选择完全不同的 Expert。

因此 Expert 不应被简单人格化成固定的“数学专家”“代码专家”。训练后可能出现一定功能分化，但路由的实际单位通常是 Layer 内的 Token 表示。

### 2.4 总参数与激活参数

MoE 模型可以拥有很多 Expert，因此总参数量很大；但每个 Token 只运行少数 Expert，所以每 Token 激活参数量小得多。

以公开 Qwen3-30B-A3B 的命名直觉为例：

```text
约 30B：总参数规模
约 3B：每个 Token 实际激活的参数规模量级
```

这不意味着整个模型只需要保存 3B 参数：所有 Expert 权重仍要存储并分布到设备上。

### 2.5 MoE 为什么让系统更复杂

Dense FFN 的 Token 通常留在当前并行布局内完成矩阵计算；MoE 中 Router 可能把 Token 发给位于其他设备的 Expert。

因此系统还要处理：

- Token Dispatch；
- Expert 所属设备；
- 跨设备 All-to-All 类通信；
- 不同 Expert 的负载不均；
- Expert 参数的本地与全局编号；
- Expert 权重的保存、转换和同步。

当前阶段只要先掌握“为什么需要通信与编号恢复”，暂时不需要研究通信 Kernel。

---

## 3. TP、EP、PP 在拆什么

### 3.1 TP：Tensor Parallel

TP 把同一个大参数矩阵或张量计算拆到多张 GPU。

例如：

```text
完整 linear_qkv.weight
├── TP Rank 0 保存/计算一部分
└── TP Rank 1 保存/计算另一部分
```

因此 `TP shard` 是：

> 同一个逻辑参数的局部分片。

把训练参数导出给 HF/vLLM 时，可能需要按正确维度 Gather、Concat 或转换布局。

### 3.2 EP：Expert Parallel

EP 把不同 Expert 放到不同设备。

假设有 8 个 Expert，EP=2：

```text
EP Rank 0：Global Expert 0、1、2、3
EP Rank 1：Global Expert 4、5、6、7
```

Router 选中 Expert 6 时，Token Hidden State 必须被发送到持有 Expert 6 的设备。

### 3.3 PP：Pipeline Parallel

PP 把不同 Transformer Layer 分到不同 Stage：

```text
PP Stage 0：Global Layer 0～23
PP Stage 1：Global Layer 24～47
```

Stage 1 内部的第一个本地 Layer 可能叫：

```text
Local Layer 0
```

但导出时它必须恢复为：

```text
Global Layer 24
```

所以需要 `PP layer offset`。

### 3.4 Local Expert 与 Global Expert

EP Rank 1 内部可能把自己持有的第一个 Expert 叫：

```text
Local Expert 0
```

它在完整模型里实际是：

```text
Global Expert 4
```

参数转换时如果忘记恢复全局编号，可能把 Expert 4 的权重错误发送给推理模型的 Expert 0。

### 3.5 DP 与以上三者的区别

Data Parallel 主要复制模型、切分数据：

```text
不同 DP Replica
→ 处理不同 Microbatch
→ 聚合梯度
```

而 TP/EP/PP 是把一个模型实例内部的计算或参数拆开。

简化记忆：

```text
DP：拆数据
TP：拆矩阵
EP：拆专家
PP：拆层
CP：拆长序列上下文
```

---

## 4. 为什么同一模型会有三种参数世界

同一逻辑模型在不同系统里可能使用不同表示：

```text
HF Checkpoint 表示
↕ Bridge / Converter
Megatron 分布式训练表示
↕ 在线转换与通信
vLLM 推理表示
```

### 4.1 HF Checkpoint

HF 是 Hugging Face。HF Checkpoint 指兼容 Transformers 生态的目录、配置和参数命名，不要求文件一定从 Hugging Face 网站下载。

典型内容：

```text
config.json
tokenizer.json / tokenizer_config.json
model.safetensors
或若干 model-xxxxx-of-xxxxx.safetensors
权重索引文件
generation_config.json
```

它主要适合：

- 发布和部署；
- 跨框架加载；
- vLLM/SGLang/Transformers 推理；
- 语义清晰的参数命名。

### 4.2 分布式 Checkpoint

分布式 Checkpoint 主要用于恢复训练，通常包含：

- 按 TP/PP/EP 等布局保存的模型分片；
- Optimizer State；
- Scheduler；
- Global Step；
- RNG State；
- 混合精度状态；
- 数据进度和并行元数据。

它不一定能被 vLLM 直接加载。

### 4.3 多个 safetensors 文件不等于 TP 分片

HF 模型可能为了避免单文件过大而拆成多个 `safetensors` 文件。这是存储切片，不自动等于训练时的 Tensor Parallel 布局。

判断依据应是：

- 参数命名；
- Index 文件；
- Checkpoint 格式；
- 加载器如何重组；
- 是否保存并行拓扑元数据。

### 4.4 在线推理权重

RL 中常不先落盘 HF Checkpoint，而是：

```text
Megatron Actor 更新
→ 按推理模型名称和 Shape 转换
→ 通过通信层在线发送
→ vLLM 原地更新运行时权重
```

所以在线同步的核心要求是：

```text
参数完整
+ 名称正确
+ Shape 正确
+ 分片还原正确
+ 版本一致
```

---

## 5. Adapter、Bridge 与 Converter

### 5.1 Model Adapter

这里的 Adapter 不是 LoRA Adapter。

Model Adapter 是模型家族接入策略，告诉通用框架：

- 怎样识别 `model_type` / `architectures`；
- 怎样修正或解释 Config；
- 应构建哪种 Layer Spec；
- Forward 需要哪些特殊输入；
- Position ID、Attention 或状态怎样处理；
- 参数名称如何映射。

它回答的是：

> “这个模型家族应该怎样接入通用训练/推理框架？”

### 5.2 Bridge / mbridge

Bridge 可以理解为 HF 世界和 Megatron 世界之间的翻译器：

```text
HF Config
→ Megatron TransformerConfig

HF 参数名与布局
↔ Megatron 融合/分片参数
```

例如 HF 可能使用：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

Megatron 可能使用融合并分片后的：

```text
linear_qkv.weight
```

Bridge 不是训练算法、Ray、推理引擎或通信协议；它是模型构建和参数格式转换层。

### 5.3 Converter

Converter 更聚焦于一条具体参数转换路径，例如：

```text
Megatron 分片参数
→ 还原全局 Layer/Expert 编号
→ 转换成 HF/vLLM 所需名称和 Shape
```

Bridge 与 Converter 在不同项目里边界可能重叠。阅读代码时不要只根据类名判断，要检查它实际负责：

- Config；
- 模型构建；
- Checkpoint 导入/导出；
- 在线同步；
- 还是以上多个环节。

### 5.4 参数 ABI 是什么

这里的 ABI 不是只指 C/C++ 二进制接口，而是借用“接口契约”的含义：

```text
参数名称
+ Shape
+ DType
+ 分片方式
+ Layer/Expert 编号
+ 顺序
+ 版本
= 训练引擎与推理引擎之间的参数契约
```

只要任何一项不一致，就可能：

- 直接报 Missing/Unexpected Key；
- Shape 不匹配；
- 权重发给错误 Layer/Expert；
- 更危险地静默加载到错误位置。

---

## 6. MTP 在模型上增加了什么

### 6.1 NTP 主路径

普通 Next-Token Prediction：

```text
位置 t 的 Hidden State
→ 预测 Token t+1
```

### 6.2 MTP 路径

MTP 让模型额外学习更远的未来 Offset：

```text
主路径     → 预测 t+1
MTP Depth 1 → 预测 t+2
MTP Depth 2 → 预测 t+3
...
```

具体 MTP 模块可能只是额外投影，也可能包含更完整的 Transformer/Attention/MoE 结构，必须以目标模型的 Config 和权重 Key 为准。

### 6.3 为什么 MoE 会放大 MTP 转换难度

如果 MTP Layer 内也包含 MoE：

```text
MTP Layer
├── Attention
├── Router
└── Expert FFN
```

那么 MTP 参数也可能涉及：

- TP 矩阵分片；
- EP Expert 分片；
- PP 所在 Stage 和 Layer Offset；
- Local/Global Expert 编号；
- 推理侧对应模块的参数名和 Shape。

所以“Converter 没覆盖 MTP”可能不是漏复制几行权重，而是缺少完整的模型结构与参数 ABI 定义。

---

## 7. 完整 MTP 支持的五个契约

### 7.1 契约一：Checkpoint 与模型构建

必须确认：

- Config 是否声明 MTP 层数或结构；
- 权重文件是否真的包含 MTP 参数；
- Bridge/Megatron 是否能构建对应模块；
- MTP-off 时是否可以选择不构建；
- MTP-on 时是否加载了正确参数，而非随机初始化后直接用于推理。

仅设置 `enable=true` 不会凭空产生训练好的 MTP 能力。

### 7.2 契约二：Future Label 与 MTP Loss

训练侧需要：

```text
构造不同未来 Offset 的 Label
→ 同步 Shift Loss Mask
→ MTP Forward
→ 计算每层 MTP Loss
→ 按权重加入 Total Loss
→ Backward
→ MTP 参数进入 Optimizer
```

模块存在但没有 Label/Loss，MTP 参数就不一定获得有效训练。

### 7.3 契约三：保存与恢复

必须分别验证：

- HF 导出是否保留 MTP Config 和参数；
- 分布式 Checkpoint 是否保存 MTP 参数；
- 续训是否恢复相应 Optimizer State；
- Save→Reload 后 MTP 数值是否一致；
- MTP-off 导出是否正确移除不应存在的字段和参数。

### 7.4 契约四：训练到推理的转换与在线同步

必须确保：

- 所有推理依赖的 MTP 参数都被枚举；
- TP/EP/PP 分片被正确还原或重分片；
- Local Layer/Expert 编号恢复为 Global 编号；
- 参数名、Shape 和 DType 符合推理端；
- vLLM 主模型和 MTP proposer 使用同一 Actor 版本。

“主模型权重同步成功”不自动证明 MTP 参数也同步成功。

### 7.5 契约五：推理侧 Draft/Verify

必须确认：

- 推理引擎支持目标模型族的 MTP；
- MTP proposer 确实被实例化；
- Draft 深度不超过 Checkpoint 能力；
- Verify、Accept/Reject 和 Cache/State 提交正确；
- 有 Acceptance Rate、Acceptance Length 和吞吐指标；
- MTP-on 的真实吞吐优于 MTP-off，而不只是功能能跑。

---

## 8. 为什么 MTP 要拆成 Load、Train、Rollout 三个开关

### 8.1 三个逻辑维度

```text
Load MTP
→ 是否构建、加载、保存和同步 MTP module

Train MTP
→ 是否构造 Future Label、计算 MTP Loss 并更新参数

Rollout MTP
→ 是否让推理引擎使用 MTP 投机解码
```

它们有关联，但不是同一件事。

### 8.2 合法模式矩阵

| 模式 | Load | Train | Rollout | 含义 |
|---|---:|---:|---:|---|
| 普通 Baseline | 否 | 否 | 否 | 普通 NTP RLVR |
| Load-only | 是 | 否 | 否 | 保留/导出 MTP，但不训练、不推测 |
| Train-only | 是 | 是 | 否 | 训练 MTP，Rollout 仍普通 Decode |
| Rollout-only | 是 | 否 | 是 | 使用已有 MTP 推测，但本轮不训练 MTP |
| Full MTP | 是 | 是 | 是 | 训练 MTP，并用于 Rollout |

依赖关系：

```text
Train MTP = true
→ Load MTP 必须为 true

Rollout MTP = true
→ Load MTP 必须为 true
→ Checkpoint 必须有有效 MTP 参数
→ 推理引擎必须支持目标模型
→ 在线同步必须覆盖 MTP 参数
```

### 8.3 Rollout-only 为什么可能逐渐失配

如果：

```text
主 Actor 持续 RL 更新
MTP Head 保持冻结
```

那么 MTP proposer 可能逐渐不适配新的主模型分布，导致 Acceptance Rate 下降。

因此 Rollout-only 可以作为特定模式存在，但不能默认认为长期稳定。

### 8.4 `detach_encoder` 在问什么

MTP Auxiliary Loss 可以有两种主要梯度边界：

```text
Detach 主干
→ MTP Loss 只训练 MTP module

不 Detach 主干
→ MTP Loss 同时影响 MTP module 和共享主干
```

前者减少对主 RL 目标的干扰；后者可能让主干也学习多 Token 预测，但会改变优化目标和回归风险。

默认值不能只凭直觉决定，必须结合：

- 目标 Checkpoint 的训练方式；
- 上游参考实现；
- 团队期望；
- MTP-off 回归和 MTP-on 效果实验。

---

## 9. 怎样保证向后兼容

### 9.1 默认关闭

新功能默认应保持：

```text
Load MTP = false
Train MTP = false
Rollout 不设置 Speculative Decoding
```

### 9.2 MTP-off 不只是“不报错”

回归标准应包括：

- 旧 YAML 无需修改即可解析；
- 不构建额外 MTP 模块；
- 不加载或同步 MTP 参数；
- 不产生 MTP Loss 和指标；
- 显存和参数数量不意外增加；
- vLLM 仍走普通 Decode；
- 普通 RLVR baseline 的结果和控制流保持一致。

### 9.3 避免重复状态源

如果训练和推理组件已经各有配置空间，更合理的是：

```text
Megatron 配置：负责 Load / Train MTP
vLLM 配置：负责 Speculative Decoding
统一 Validator：检查组合是否合法
```

不要在两个地方重复维护相同的 `enable_rollout` 状态，否则可能出现一边开、一边关的冲突。

---

## 10. 通用逻辑与模型特定逻辑怎样分层

```text
通用配置层
├── Load/Train 开关
├── 组合校验
└── 默认关闭与旧配置兼容

通用训练层
├── Future Label
├── Loss Mask Shift
├── MTP Loss / Scaling
└── MTP 指标

通用 Rollout 层
├── Speculative Config
├── Acceptance Metrics
└── Pause / Sync / Resume 协议

模型家族 Adapter
├── Config 字段识别
├── MTP Module 构建
├── Checkpoint Key
├── 参数名与 Shape 映射
├── TP/EP/PP 分片还原
└── 推理引擎模型 ABI
```

这样适配下一个模型家族时：

- Label、Loss、指标和通用校验可以复用；
- 主要新增模型结构、参数映射和推理兼容性；
- 如果新模型还有 GDN/混合 Attention，则另加状态和并行约束。

---

## 11. 静态代码审计与运行验证必须分开

### 11.1 静态审计可以证明什么

- 是否存在配置字段；
- 是否构建某类模块；
- Forward 是否传入 MTP Label；
- Converter 是否有 MTP 参数分支；
- 某组合是否显式抛未实现错误；
- 推理配置是否能够透传。

### 11.2 静态审计不能直接证明什么

- 目标 Checkpoint 真的包含哪些参数；
- MTP Loss 数值正确；
- 参数真的获得梯度并被 Optimizer 更新；
- 保存/恢复 Round Trip 正确；
- vLLM 真的创建了 MTP proposer；
- 在线同步没有漏参数；
- Acceptance 和吞吐有收益。

因此能力矩阵的每一格都应标记证据类型：

```text
配置存在
静态代码覆盖
单元测试通过
单卡运行通过
多卡运行通过
端到端训练通过
端到端推理收益验证
```

不要把“有骨架”写成“已支持”。

---

## 12. MTP 适配的推荐验收顺序

```text
0. MTP-off 回归
   旧配置、显存、参数数目和普通 Decode 不变

1. Checkpoint Audit
   Config 字段、MTP Key、层数、Shape 明确

2. Build-only
   MTP Module 能正确构建和加载

3. Train-only
   Future Label、MTP Loss、Gradient、Optimizer 更新正确

4. Save/Reload
   HF/分布式 Checkpoint Round Trip 正确

5. Sync-only
   参数名、Shape、TP/EP/PP 和版本映射正确

6. Rollout-only
   proposer 实例化、Draft/Verify、Acceptance 指标正确

7. Full MTP Smoke
   RL Update 与 MTP Rollout 闭环完成

8. Correctness / Performance A-B
   MTP-off 与 MTP-on 分桶比较结果和吞吐
```

这个顺序的价值是：每一步只跨越一个新的接口边界，失败时容易定位。

---

## 13. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| MoE 为整个 Prompt 选择一个固定专家 | Router 通常逐 Token、逐 Layer 选择 Top-K Expert |
| Expert 一定能被命名成数学/代码专家 | 可能形成分工，但路由依据是当前层 Token Hidden State |
| 30B-A3B 表示模型只保存 3B 参数 | 总参数仍约 30B，只是每 Token 激活量约 3B |
| MoE 改变的是 Attention | 常见 MoE 主要替换 FFN；Attention 可以仍是 GQA |
| TP、EP、PP 都只是把模型复制多份 | 它们分别拆矩阵、专家和层；DP 才主要复制模型、拆数据 |
| Local Expert 0 就是全局 Expert 0 | 不同 EP Rank 的 Local 0 对应不同 Global Expert |
| 多个 safetensors 文件就是 TP 分片 | 可能只是文件存储切片，与训练拓扑不同 |
| Adapter 就是 LoRA | 这里是模型家族接入策略，不是参数高效微调模块 |
| Bridge 是通信库 | 它主要翻译 Config、模型结构和权重格式 |
| 配置里有 MTP 开关就算支持 MTP | 还需 Label/Loss、保存、同步、推理和收益验证 |
| MTP 参数存在就一定会训练 | 还必须参与 Forward/Loss、获得梯度并进入 Optimizer |
| 主模型同步成功就代表 MTP 同步成功 | Converter 可能只覆盖主干，MTP 参数需要单独验证 |
| MTP-on 能运行就说明有加速 | 还需测 Acceptance、Draft/Verify 成本和稳态吞吐 |
| 支持 Qwen3 30B 就自动支持 Qwen3.5 | 模型结构、GDN、状态、并行和参数 ABI 可能不同 |

---

## 14. 一分钟复习

1. Dense Layer 的每个 Token 使用同一 FFN；MoE Layer 的 Router 为每个 Token 选择少数 Expert FFN。
2. TP 拆矩阵，EP 拆 Expert，PP 拆 Layer，DP 拆数据。
3. Local Layer/Expert 编号必须结合 PP/EP Offset 恢复成全局编号。
4. HF Checkpoint 面向加载部署；分布式 Checkpoint 面向恢复训练；在线同步把 Megatron 权重直接转换给 vLLM。
5. Adapter 决定模型家族怎样接入；Bridge 翻译 HF 与 Megatron；Converter 落实具体参数映射。
6. MTP 完整支持需要构建、Loss、保存、同步和推理五个契约。
7. Load、Train、Rollout MTP 应是可校验的独立语义，不能只放一个总开关。
8. MTP-off 必须保持旧行为，而不只是“不报错”。

---

## 15. 自测问题

### 问题 1

MoE 为什么不是“整个数学 Prompt 都送给数学专家”？

期望回答：Router 通常在每个 MoE Layer 中根据每个 Token 当前的 Hidden State 选择 Top-K Expert；不同 Token、不同 Layer 的选择都可能不同。

### 问题 2

TP、EP、PP 各自切分什么？

期望回答：TP 切同一矩阵/张量，EP 切不同 Expert，PP 切不同 Layer。

### 问题 3

为什么 EP Rank 1 的 Local Expert 0 可能是 Global Expert 4？

期望回答：每个 Rank 用局部连续编号管理自己持有的 Expert；导出完整模型时必须加 EP Offset 恢复全局编号。

### 问题 4

HF Checkpoint 与分布式 Checkpoint 的主要用途差异是什么？

期望回答：HF 格式偏发布、部署和跨框架加载；分布式格式偏训练恢复，通常保留并行分片和 Optimizer/Step/RNG 等状态。

### 问题 5

为什么 MTP Module 已经构建出来，仍不代表它会学习？

期望回答：还需要 Future Label、MTP Loss、计算图连接、梯度、Optimizer Parameter Group 和实际 Step。

### 问题 6

为什么主模型在线同步成功，仍不能证明 MTP Rollout 使用了新参数？

期望回答：Converter 可能漏掉 MTP 参数或分片/编号映射错误；还需检查 MTP 参数覆盖、版本和推理端输出。

### 问题 7

Train-only 与 Rollout-only MTP 有什么区别？

期望回答：Train-only 计算并更新 MTP Loss，但推理仍普通 Decode；Rollout-only 使用已有 MTP proposer 投机解码，但本轮不通过 MTP Loss 更新它。

### 问题 8

为什么 `detach_encoder` 会改变验收标准？

期望回答：它决定 MTP Auxiliary Loss 只更新 MTP Head，还是也影响共享主干；两者训练目标、梯度覆盖和回归风险不同。

### 问题 9

怎样证明 MTP-off 真正向后兼容？

期望回答：旧配置无需修改，MTP Module/Loss/同步/推理解码都不出现，显存与参数数量不意外变化，普通 baseline 行为保持一致。

---

## 16. 当前任务下的最小学习边界

现在必须掌握：

1. Transformer Layer = Attention + FFN；
2. MoE 替换普通 FFN，Router 逐 Token、逐 Layer 路由；
3. TP/EP/PP 与 Local/Global 编号；
4. HF、Megatron、vLLM 三种参数表示；
5. Adapter、Bridge、Converter 的职责；
6. MTP 的构建、训练、保存、同步和推理五个契约；
7. MTP-off/Train-only/Rollout-only/Full MTP 模式矩阵。

当前可以延期：

- Router 梯度公式；
- Load Balancing Loss；
- All-to-All Kernel；
- Grouped GEMM；
- EP 性能优化；
- 复杂 CP/GDN 耦合；
- PPO/GRPO 完整数学推导。

---

## 17. 公开参考入口

- [vLLM：MTP Speculative Decoding](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md)
- [AReaL PR #1445：MTP-Augmented SFT/RL Training](https://github.com/areal-project/AReaL/pull/1445)
- [AReaL PR #1403：MTP Head Opt-in](https://github.com/areal-project/AReaL/pull/1403)
- [verl PR #4936：MTP 训推设计参考](https://github.com/verl-project/verl/pull/4936)

阅读这些参考时，要区分：

```text
已合并的稳定能力
Draft/Blocked PR 中的设计参考
特定模型/后端实现
可以抽象复用的通用契约
```

不要把其他框架或其他模型的补丁机械复制成当前模型的最终实现。
