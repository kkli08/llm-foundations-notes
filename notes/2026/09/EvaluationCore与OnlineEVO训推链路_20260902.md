# Evaluation Core 与 Online EVO 训推链路

> 日期：2026-09-02
>
> 来源：近期 Online EVO、Agent Rollout、Sandbox 与 Evaluation 架构调研
>
> 状态：已整理；具体工程 Schema 仍待目标系统确认

## 0. 这篇笔记解决什么问题

最容易画错的链路是：

```text
Rollout → Online Eval → Train
```

它把“训练样本生成”和“候选版本评测”误画成了必经的串行阶段。

更准确的关系是：

```text
训练主链：
TRAIN_ROLLOUT（内部可含 Reward/Judge）
→ Advantage
→ PPO/GRPO Update
→ Weight Sync

独立评测支线：
某个 Candidate/Policy/Skill 版本
→ EVAL_ROLLOUT
→ Score / Reduce / Metric
→ EvalResult
→ 训练决策、EVO 决策、发布门禁或报告
```

## 1. Eval 本身不分 Online / Offline

Evaluation Core 应是一套可复用能力。Online/Offline 的区别主要是 Trigger 和结果消费方式。

### Online Eval

```text
训练 Step / 新 Candidate / 新 Skill 版本事件
→ 触发 Eval Job
→ 结果立即进入训练、EVO 或发布闭环
```

### Offline Eval

```text
独立脚本 / 定时批任务 / 人工请求
→ 触发相同 Eval Core
→ Benchmark / 回归 / 报告
```

两者应尽量复用：

```text
Task
Executor
Scorer
Reducer
Metric
EvalLog / Provenance
```

所以“Online Evaluation Service”不应重新发明一套与 Offline Benchmark 完全不同的评分语义。

## 2. TRAIN_ROLLOUT 与 EVAL_ROLLOUT 是用途，不只是名称

### 2.1 TRAIN_ROLLOUT

输出可能包括：

- Token IDs；
- Behavior Logprobs；
- Reward/Judge 结果；
- Loss Mask；
- Policy/Weight Version；
- Tool/Environment Trajectory。

它会进入 Advantage、Logprob 重算和 Trainer。

### 2.2 EVAL_ROLLOUT

输出主要进入：

- Scorer；
- Attempt Reducer；
- Metric Aggregator；
- EVO Candidate Selection；
- 报告或发布门禁。

它不应自动进入训练 Batch。

### 2.3 为什么必须显式隔离

如果现有 Session 在结束时默认把所有 Trajectory 导出为训练数据，新 Eval Executor 复用这条路径时可能把 Held-out Eval 样本混进训练集，造成：

- 数据泄漏；
- Benchmark 污染；
- Advantage 输入混乱；
- 对候选版本过拟合。

推荐让每个请求带：

```text
purpose = TRAIN_ROLLOUT | EVAL_ROLLOUT
```

并在 Storage、Queue、Gateway 和 Sink 全链路保持该字段，或直接使用物理隔离的 Eval Executor/Sink。

## 3. Evaluation Core 的对象边界

推荐数据流：

```text
CandidateRef + TaskSet / TaskSpec + EvalSpec
→ AttemptRequest
→ RolloutExecutor
→ EpisodeResult
→ Scorer
→ AttemptReducer
→ MetricAggregator
→ EvalResult
```

### 3.1 CandidateRef

表示“评测谁”：

- Model/Policy Version；
- Skill/Prompt/Workflow Version；
- Config Digest；
- Artifact Reference。

### 3.2 TaskSpec

表示运行前的任务定义：

- Task ID；
- 输入与环境需求；
- 允许的工具；
- Timeout；
- Judge/Scorer 配置引用。

### 3.3 EvalSpec

表示怎样评：

- Task Set；
- Repeats/Seeds；
- Executor；
- Scorer；
- Reducer；
- Metric；
- 并发与失败策略。

### 3.4 AttemptRequest

一条具体执行请求：

```text
Candidate × Task × Repeat × Seed
```

### 3.5 EpisodeResult

表示运行后事实：

- 完整或裁剪后的 Trajectory；
- 终态与 Failure Category；
- Raw Judge Output；
- Token/Latency/Tool Metadata；
- 实际使用的版本。

`TaskSpec` 是运行前定义，`EpisodeResult` 是运行后事实。不要在同一个对象中不断修改“计划、运行态和结果”，否则 Retry、Replay 和审计都变得困难。

## 4. Provenance 为什么是第一等公民

至少记录：

```text
run_id
candidate_id / version / digest
task_id
attempt_id / repeat_index
seed
policy / skill / harness version
executor version
scorer / reducer version
```

没有这些字段，就无法回答：

- 这条轨迹由哪个候选生成？
- 重跑为何不同？
- Scorer 变化还是 Candidate 变化？
- 失败来自 Agent、环境还是基础设施？
- 哪些 Attempt 能被安全聚合？

## 5. RolloutExecutor 必须可插拔

Evaluation Core 不应绑定某个 Agent Sim 或 Sandbox：

```text
EvaluationCore
→ RolloutExecutor
   ├── NativeTrainingRolloutExecutor
   ├── AgentSimRolloutExecutor
   ├── RemoteSandboxRolloutExecutor
   └── Replay / MockExecutor
```

单次执行也属于 Evaluation。`task × repeats` 是 EvalSpec 配置，不应把“多次 Agent Sim”写成 Evaluation 的定义。

统一 Executor 接口至少要覆盖：

- Submit；
- Poll/Stream Events；
- Cancel；
- Timeout；
- Idempotency Key；
- EpisodeResult；
- Failure Category。

## 6. Trigger、Orchestration 与 Execution 是三层

以常见训练框架为例，可以拆成：

```text
Evaluator / Trigger
→ 判断 Step、Epoch、Time 或事件是否到达

Evaluation Orchestrator
→ 遍历 TaskSet，生成 AttemptRequest，聚合 Job 状态

Rollout Controller / Executor
→ 真正并发执行 Episode
```

只看到一个 `evaluator.py` 文件，不能就把它当成完整 Evaluation Service。Online EVO 还通常需要：

- Candidate Version；
- 异步 Job 生命周期；
- 多 Executor；
- Partial Success；
- Retry/Cancel；
- Scorer/Reducer/Metric；
- Result Query；
- 发布门禁。

## 7. Agent Sim / Sandbox 接入后，谁发生了变化

### 7.1 传统 Trainer 内部 Workflow

```text
Trainer
→ 取 Task
→ 调用 Policy
→ 驱动 Tool/Environment
→ 生成 Reward/Trajectory
```

### 7.2 外部 Agent Sim 驱动

```text
Agent Sim
→ 拉取 Task
→ 申请/连接 Sandbox
→ 驱动 Agent Loop
→ 通过 Gateway 调用当前 Policy
→ 回传完整 Trajectory
```

变化的是 Episode 的驱动方。以下能力仍留在训练侧：

```text
Advantage
→ Actor Logprob / Reference
→ Megatron Update
→ Weight Sync
```

因此“把 Rollout 交给外部 Agent Sim”不等于把 PPO/GRPO Trainer 也移出训练框架。

## 8. 两种 Model Serve 不要混淆

### Environment / Sandbox Serve

负责：

- 场景环境分配；
- 隔离与健康；
- Lease/Session；
- 回收与重建。

### Policy Inference Serve

例如 SGLang/vLLM，负责：

- 加载当前 Policy；
- Token Generation；
- Logprob；
- KV/State；
- Weight Version 更新。

二者都可能被口头简称为 Model Serve，但它们服务的对象完全不同。

若外部 Agent 直接调用 Policy Server，仅返回自然语言文本通常不够。训练 Rollout 还可能要求：

```text
Token IDs
Behavior Logprobs
Policy / Weight Version
Sampling Metadata
Loss Mask / Turn Boundary
```

## 9. EVO Controller 与 Evaluation Core 的关系

Evaluation Core 是通用能力；EVO Controller 是调用方：

```text
Candidate v1
→ EvaluationCore
→ EvalResult
→ Failure Attribution / Candidate Generator
→ Candidate v2/v3
→ EvaluationCore
→ Compare / Accept / Reject
→ Version Publish
```

可演化对象可能包括：

- System Prompt；
- Skill；
- Tool Config；
- Workflow；
- Sampling Config；
- Context Organization。

这些对象不应被硬编码进 Eval Core。Eval Core 只认 CandidateRef 和执行/评分协议。

## 10. Failure 与 Partial Success

一个 Eval Job 包含多个 Attempt：

```text
Job
├── Attempt A: Success
├── Attempt B: Task Failure
├── Attempt C: Infra Failure → Retry Success
└── Attempt D: Timeout
```

需要分别定义：

- Attempt 级终态；
- Job 级完成条件；
- Infra Retry Budget；
- Task Failure 是否计入 Metric；
- Partial Success 是否允许发布结果；
- Cancel 后在途 Attempt 如何 Drain；
- 重复回调如何幂等。

基础设施失败不能直接写成低 Reward；否则 EVO 会把系统故障错误归因给 Candidate。

## 11. 可以怎样借鉴公开框架

| 参考方向 | 借鉴重点 |
|---|---|
| Inspect AI | Eval Core、Scorer/Reducer/Metric、EvalLog、动态 Task Source |
| lm-eval-harness | Task 编译为 Typed Request 与批量分发 |
| DeepEval | 运行前 Golden/Task 与运行后 Episode/TestCase 分离 |
| GEPA | Candidate 与 Evaluation 的协议 |
| Agent Lightning | 异步 Rollout 执行面与事件采集 |
| Skill 优化框架 | Candidate 接受、拒绝、验证与发布门禁 |

新开发分支中的动态接口适合作为设计参考，但不应在没有稳定性评估时直接成为生产硬依赖。

## 12. 最小实现顺序

1. 定义 `CandidateRef / TaskSpec / EvalSpec / AttemptRequest`；
2. 定义 `EpisodeResult / FailureInfo / EvalResult`；
3. 用 MockExecutor 验证单 Attempt 与 Retry；
4. 增加 AttemptReducer、MetricAggregator 和 Provenance；
5. 增加异步 Eval Job、Cancel、Partial Success 和幂等；
6. 接一个真实 RolloutExecutor；
7. 加 `purpose=TRAIN|EVAL` 的全链隔离测试；
8. 再接 EVO Candidate Loop；
9. 最后接 DatasetExecutionRouter 和正式入口。

## 13. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| Rollout 后必须先 Online Eval 才能训练 | TRAIN_ROLLOUT 可直接进入 Reward/Advantage/Update |
| Online Eval 和 Offline Eval 是两套核心 | 主要差异是 Trigger 与结果消费，Core 应复用 |
| 训练和评测共用 Runtime 就会自动正确 | 必须隔离 Purpose、Storage、Sink 与数据集 |
| Evaluation 就是多跑几次 Agent Sim | 单次也属于 Eval；Executor 和 Repeats 都是配置 |
| `evaluator.py` 就是完整 Evaluation Service | 还缺 Job、Executor、Failure、Reducer、Metric、Result Query |
| 外部 Agent Sim 接管了整个 RL | 它接管 Episode 驱动；Advantage/Update/Sync 仍在 Trainer |
| Sandbox Model Serve 和 vLLM Model Serve 是一回事 | 一个服务环境，一个服务 Policy Token Generation |
| Harness Optimization 应写进 Eval Core | 它是 Candidate 生成/优化方，Eval Core 只评候选 |

## 14. 一分钟复习

```text
训练主链：
TRAIN_ROLLOUT → Reward/Advantage → PPO/GRPO Update → Weight Sync

评测支线：
Candidate + Task + EvalSpec
→ EVAL_ROLLOUT
→ EpisodeResult
→ Score → Reduce → Metric
→ EvalResult

Online / Offline 主要改变 Trigger；
TRAIN / EVAL 必须隔离数据用途。
```

## 15. 自测问题

1. 为什么 `Rollout → Online Eval → Train` 不是通用正确链路？
2. Online 与 Offline Eval 应复用哪些核心对象？
3. TRAIN_ROLLOUT 与 EVAL_ROLLOUT 共用 Executor 时，为什么仍必须显式隔离？
4. TaskSpec 与 EpisodeResult 为什么不能混成一个不断变异的对象？
5. 外部 Agent Sim 接管 Rollout 后，哪些 RL 训练能力仍留在 Trainer？
6. Sandbox Serve 和 Policy Inference Serve 分别服务什么？

## 16. 与已有知识的联系

- [Online EVO Rollout 架构与任务边界](../08/OnlineEVO_Rollout架构与任务边界_20260828.md)：旧文建立 Task→Trajectory 与 EVO/Environment 分层；本文新增 Evaluation Core、TRAIN/EVAL Purpose 和 Candidate 协议。
- [Online RL 的 Cohort 并发与会话容量](../08/OnlineRL的Cohort并发与会话容量_20260818.md)：Eval Job 同样需要并发容量、长尾、Timeout 和 Session 生命周期管理。
- [On-policy 训推与 MTP 性能分析](../08/OnPolicy训推与MTP性能分析_20260806.md)：TRAIN_ROLLOUT 的 Policy Version、Behavior Logprob 与新鲜度仍遵守 On-policy 契约。
- [MTP 模型状态流与在线权重事务](../07/MTP模型状态流与在线权重事务_20260729.md)：CandidateRef 和 EvalResult 需要继承 Version、Manifest 与事务提交思想。
