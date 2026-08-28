# Online EVO Rollout 架构与任务边界

> 日期：2026-08-28
>
> 来源：近期 Online EVO Rollout 需求梳理与开源框架调研
>
> 状态：已整理；具体业务协议仍待目标工程确认

## 0. 这篇笔记解决什么问题

“EVO Online Rollout Infra”不是业界统一名词。当前最容易混淆的是：EVO 到底在演化什么、Rollout 为什么要独立成基础设施、它和 Online RL Rollout、Sandbox 资源调度是什么关系。

先用一句话概括：

> Online EVO Rollout 是一条可并发执行 `Task → Trajectory` 的在线轨迹生产路径；它把当前 Agent/Skill/Prompt 放进环境执行，采集结果供后续失败归因、Skill 演化和回放验证使用。

## 1. 用一个具体例子理解 EVO

假设 Agent 有一条 Skill：

```text
“进入房间后，先观察，再寻找目标物品，最后执行交互。”
```

它在 20 个新任务上 Rollout，其中 8 个失败。EVO 流程可能做：

```text
8 条失败 Trajectory
→ 区分环境故障、Agent 没遵循 Skill、Skill 本身有缺陷
→ 总结可复用失败模式
→ 生成 Skill v2 候选
→ 用未参与归纳的任务重新 Rollout
→ 比较 v1/v2
→ 只有验证通过才发布 v2
```

这里真正“演化”的通常是：

- Skill；
- Prompt；
- Agent Workflow；
- 工具使用策略；
- 某种可版本化的行为配置。

它不一定会修改模型权重，所以不能把 EVO 直接等同于 RL 训练。

## 2. 三层必须分开

### 2.1 EVO 业务层：从轨迹中学什么

```text
Trajectory
→ Failure Attribution / Reflection
→ 生成修改候选
→ Held-out Replay Validation
→ 版本选择与发布
```

这层回答“怎样从成败经验改进 Agent”。

### 2.2 Rollout 执行层：怎样可靠地产生轨迹

```text
Task
→ 排队与并发
→ Agent 执行
→ Tool/Environment 交互
→ Timeout / Cancel / Retry
→ Trajectory + Result + Status
```

这层回答“怎样把很多任务可靠地跑完并记录下来”。

### 2.3 Environment/Sandbox 层：任务在哪里跑

它负责隔离环境、资源生命周期、健康检查和后端适配。Rollout 层应通过接口使用环境，而不是把某个具体 Sandbox 的创建命令写进核心业务逻辑。

三层关系：

```text
EVO Processor
      ↑ Trajectory
Rollout Controller / Executor
      ↓ Execution Request
Environment Adapter
      ↓
Local / Container / Sandbox / Simulator
```

## 3. Online 在这里是什么意思

Online EVO 通常表示：系统在当前模型、Agent、Skill 或配置下持续生成新轨迹，并把新结果反馈给演化循环。

但它不自动等于严格 On-policy RL：

- On-policy RL 强调 Behavior Policy 与 Current Policy 的版本关系；
- EVO 可能只修改 Skill/Prompt，而不做 Policy Gradient；
- EVO 轨迹也可能进入离线分析或 replay validation，而不是直接进 Optimizer。

因此记录时应分别写清：

```text
谁生成轨迹？
演化对象是什么？
轨迹用于更新权重、更新 Skill，还是只做评测？
新版本何时对后续 Rollout 可见？
```

## 4. 最小核心契约：Task → Trajectory

一套通用 Rollout Infra 最值得先稳定的不是某个模型调用，而是输入输出协议。

### 4.1 Task

至少应描述：

- Task ID 与 Dataset/Scene Metadata；
- Agent/Skill/Prompt 版本；
- 模型与采样配置引用；
- Environment 需求；
- Timeout、重试和取消策略；
- Trace/Correlation ID。

### 4.2 Trajectory

至少应描述：

- 按时间排序的 Observation、Action、Tool Call 与结果；
- 终态：Success、Task Failure、Infra Failure、Timeout、Cancelled；
- Reward、Judge 或业务结果（若该任务有）；
- 使用的 Agent/Skill/Model 版本；
- 时延与资源元数据；
- 可重放所需的最小输入引用。

关键原则是：基础设施失败不能伪装成低 Reward，否则演化器会把系统故障误学成 Agent 行为问题。

## 5. 为什么需要 DatasetRouter

当统一入口同时服务 EVO 数据和正式训练数据时，路由应是独立组件：

```text
Dataset Metadata
→ DatasetRouter
├── EVO Dataset → EvoRolloutExecutor
└── Formal Training Dataset → FormalRolloutExecutor
```

推荐规则：

1. 优先读取权威 Metadata/Schema，不按文件名模糊猜测；
2. 路由规则配置化并带版本；
3. 无法识别时 Fail Closed，不静默落入默认路径；
4. 支持显式 Override，便于调试；
5. 每次记录“命中的规则、输入类型、目标 Executor”；
6. Router 只决定去哪条路径，不承担具体 Rollout 业务。

## 6. 一个适合落地的模块图

```text
DatasetRouter
  ↓
RolloutController
  ├── FormalTrainingRollout
  └── EvoRollout
        ↓
AgentHarness
        ↓
EnvironmentAdapter
  ├── Mock / Local
  └── Sandbox / Simulator
        ↓
TrajectoryStore
  ├── TrainerSink
  └── EvoSink
        ↓
Failure Attribution
→ Skill/Prompt Candidates
→ Replay Validation
→ Version Registry
```

各模块只回答一类问题：

| 模块 | 核心职责 |
|---|---|
| DatasetRouter | 数据应该走哪条执行路径 |
| RolloutController | 生命周期、排队、并发、取消和状态 |
| AgentHarness | 怎样调用具体 Agent/Workflow |
| EnvironmentAdapter | 怎样获得并操作执行环境 |
| TrajectoryStore | 怎样持久化、查询和重放轨迹 |
| Evo Processor | 怎样归因失败并产生演化候选 |
| Version Registry | 哪个 Skill/Prompt 版本正在使用、何时提交新版本 |

## 7. 并发执行时最容易出错的地方

### 7.1 状态机必须明确

```text
Pending → Allocating → Running → Finalizing → Succeeded
                               ├── TaskFailed
                               ├── InfraFailed
                               ├── TimedOut
                               └── Cancelled
```

状态要幂等：重复完成回调、网络重试或 Worker 重启不能让同一 Task 被重复提交两次。

### 7.2 Retry 不能不分失败类型

- Infra Failure：环境不可用、网络中断，可在预算内重试；
- Task Failure：Agent 正常执行但没完成任务，通常是有效负样本；
- Invalid Input：Schema 或路由错误，应快速失败；
- Timeout：需要区分环境卡死、Agent 无进展和正常长任务。

### 7.3 资源与业务结果要分开提交

```text
保存 Trajectory/Result
→ 标记业务终态
→ 释放或归还 Environment
```

清理失败不应抹掉已经完成的业务结果，但必须进入可观察的资源回收队列。

## 8. 版本与正确性

EVO 是一个版本化闭环：

```text
Skill v1 生成 Trajectory
→ 产生候选 v2
→ Held-out Validation
→ Commit v2
→ 新 Rollout 使用 v2
```

每条 Trajectory 都应记录实际使用的版本，不能只记录“当前最新版本”。提交新版本最好采用事务语义：

```text
Prepare → Validate → Commit → 对新任务可见
```

已在运行的任务是否继续使用旧版本，需要明确策略；不能让一条 Trajectory 中途混用两个版本。

## 9. 如何借鉴开源项目

目前没有一个项目同时完美覆盖 EVO 语义、通用 Rollout、生产调度和具身环境。更适合按模块借鉴：

| 项目 | 最值得看的部分 |
|---|---|
| [EmbodiSkill](https://github.com/air-embodied-brain/EmbodiSkill) | 多 Epoch 轨迹、Skill-aware reflection、Skill 演化 |
| [Agent Lightning](https://github.com/microsoft/agent-lightning) | Rollout Controller、Gateway、事件/轨迹、Runner 抽象 |
| [NanoRollout](https://github.com/cocoa-org/NanoRollout) | 简洁的 Task→Trajectory、Agent Harness、Environment Backend |
| [RLinf](https://github.com/RLinf/RLinf) | 具身环境与分布式 Rollout |
| [AReaL](https://github.com/areal-project/AReaL) | Agent、Inference、Trainer、Weight Update 解耦的异步流水线 |
| [AgentRL](https://github.com/THUDM/AgentRL) | Environment Controller/Worker、Task Manager、Trajectory Buffer |
| [EvoAgentX](https://github.com/ANative-Lab/EvoAgentX) | Evaluator、Workflow Optimizer 与 Self-evolution Engine |

阅读顺序建议：

```text
EmbodiSkill：先理解演化循环
→ Agent Lightning：理解控制面和事件边界
→ NanoRollout：理解最小通用 API
→ RLinf/AReaL：再看规模化与异步
```

## 10. 最小实现顺序

1. 获取一份 EVO 数据样例和一份正式训练数据样例；
2. 定义权威 Metadata 与 `DatasetRouter`；
3. 定义 Task、Trajectory、Status、Failure Category；
4. 用 Mock Environment 跑通单任务；
5. 增加并发、Timeout、Cancel 和幂等完成；
6. 接 Trajectory Store 与基础可观测性；
7. 再接真实 AgentHarness/EnvironmentAdapter；
8. 最后接 Failure Attribution、Candidate Evolution 和 Replay Validation。

这样可以在外部 Sandbox 或正式训练链路尚未完全就绪时独立验证核心执行器。

## 11. 当前仍待工程确认的问题

- EVO 的准确演化对象是什么；
- EVO 数据与正式训练数据的权威区分字段；
- Trajectory 的正式 Schema 与存储后端；
- 是否必须产生 Reward/Judge；
- 并发、Timeout、Retry 和取消的验收口径；
- 是否复用现有 Gateway、Session 或 Scheduler；
- 最小交付是单任务、批量并发，还是 Trainer 端到端接入；
- Skill/Prompt 新版本的验证集与发布门禁。

这些是需求缺口，不应由实现者静默猜测。

## 12. 常见误解纠正

| 容易误解为 | 更准确的理解 |
|---|---|
| EVO Rollout 就是 RL Rollout | 都产生轨迹，但 EVO 可能更新 Skill/Prompt，不一定更新模型权重 |
| Online 一定等于 On-policy | Online 只说明持续生成新轨迹；On-policy 还要求策略版本关系 |
| Rollout Infra 就是 Sandbox | Sandbox 是环境后端；Rollout 还负责任务与轨迹生命周期 |
| DatasetRouter 应按数据集名称猜 | 应依赖权威 Metadata，未知类型 Fail Closed |
| 任何失败都可以当低 Reward | Infra Failure 不能污染行为学习样本 |
| 并发就是多开几个协程 | 还要处理幂等、取消、资源回收和版本一致性 |
| 有轨迹就能演化 | 还需要失败归因、候选生成和隔离验证 |

## 13. 一分钟复习

```text
Online EVO Rollout：
当前 Agent/Skill/Prompt
→ 批量执行 Task
→ 采集 Trajectory
→ 区分 Task Failure 与 Infra Failure
→ Failure Attribution
→ 生成演化候选
→ Held-out Replay Validation
→ 提交新版本
```

```text
系统分层：
EVO 业务层：从轨迹中学什么
Rollout 层：怎样可靠地产生轨迹
Environment 层：任务在哪里运行
```

## 14. 自测问题

1. 为什么 Online EVO Rollout 不等于 On-policy RL？
2. DatasetRouter 为什么必须 Fail Closed，并记录命中的规则？
3. 为什么 Infra Failure 不能被写成低 Reward？
4. AgentHarness 与 EnvironmentAdapter 分别隔离什么变化？
5. 一条 Trajectory 为什么必须记录实际 Skill/Model 版本？
6. 单任务跑通以后，并发执行器还需要补哪些正确性能力？

## 15. 与已有知识的联系

- [Online RL 的 Cohort 并发与会话容量](OnlineRL的Cohort并发与会话容量_20260818.md)：可复用其中的并发容量、长尾和 Session 生命周期思想，但 EVO 不一定使用 Cohort 或 Policy Gradient。
- [On-policy 训推、Logprob 与 MTP 性能分析](OnPolicy训推与MTP性能分析_20260806.md)：用于区分 Online 与 On-policy，并理解 Behavior/Current Policy 版本。
- [MTP 模型状态流与在线权重事务](../07/MTP模型状态流与在线权重事务_20260729.md)：版本提交、Fail-closed 和 Trajectory 实际版本记录的思想可以迁移到 Skill/Prompt 发布。
