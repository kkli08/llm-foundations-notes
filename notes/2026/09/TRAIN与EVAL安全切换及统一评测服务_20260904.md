# TRAIN 与 EVAL 安全切换及统一评测服务

> 内容日期：2026-09-04；补录时间：2026-09-07（北京时间）
>
> 来源：Online EVO、Agent Scaffold 与 Evaluation Service 设计讨论。
>
> 状态：通用设计理解；不代表具体系统已实现或通过端到端测试。

## 0. 从你的 Database/Runtime 背景理解

可以把这件事理解成：**停止接收新事务，固定读版本，让已准入的工作完成，再安全恢复写入。**但这里一条 Agent task 包含多轮模型请求和 Sandbox 资源，不能只统计当前正在 GPU 上运行的请求。

与 [9 月 3 日准入笔记](Rollout会话准入与阶段并发契约_20260903.md) 的差别是：严格“TRAIN 全部排空→EVAL”只是兼容方案；满足版本和资源条件时，可以让已准入 TRAIN tail 与 EVAL 重叠。

## 1. Task、Session、Request 和 Cohort 是四个粒度

| 对象 | 直观理解 | 可能的状态 |
|---|---|---|
| Task/job | 做完一道完整任务 | 已准入，但还在等 Sandbox |
| Session | 一次多轮 Agent 会话 | 等待工具返回，暂时没有模型请求 |
| Model request | 当前一轮生成 | GPU 执行或排队 |
| Cohort | 可供训练消费的一组轨迹 | 已完整，但尚未被 Trainer 消费 |

例子：一个 Agent 先问模型，运行工具十秒，再问模型。中间十秒 `active_model_requests=0`，但 task/session/lease 都没有结束。

所以“模型当前空闲”“服务 health 返回成功”“整个执行面已排空”是三个不同判断。

## 2. 两个门槛不要混在一起

### 进入 EVAL：准入与版本 fence

Fence 可以理解为一条原子的截止线：登记哪些 TRAIN task 已经被接受，之后拒绝新的 TRAIN task。

进入 EVAL 前需要：

1. 关闭新 TRAIN task 准入，拿到 fence 前合法 reservation 的集合/水位。
2. 冻结未分配 TRAIN capacity，保留已准入任务完成所需的资格。
3. 暂停新的参数更新和推理权重发布，把评测模型固定为 V。
4. 确认 Gateway 会按不可变 purpose、phase、version 校验和分流。
5. 确认版本兼容及资源能支持重叠，否则先排空 TRAIN。

### 恢复 TRAIN：执行面 quiescence

Quiescence 指本轮相关执行活动已静止，不再有未完成任务或悬挂资源。

恢复前需要关闭新 EVAL，收口 EVAL 与旧 TRAIN tail 的 task、session、模型请求、Sandbox lease，以及尚未交付的终态结果。失败/超时也必须有明确终态和资源释放记录。

**Ready TRAIN cohort 可以还留在队列中。**它已经不是在途执行；恢复后是否接纳，由训练侧按原 policy version、phase 和 staleness 规则决定，不能无条件视作新鲜 on-policy 数据。

## 3. 一个具体时间线

```text
t0：TRAIN-A 已登记 reservation，还在等 Sandbox
t1：关闭新 TRAIN；返回 fence_ack
t2：TRAIN-B 才来申请，没有旧 reservation → 拒绝
t3：固定评测版本 V，开放 EVAL
t4：TRAIN-A 迟到的 start_session 携旧 reservation → 校验后可继续
    EVAL-C 同时运行；两者结果进入不同出口
t5：关闭 EVAL 准入，等 A/C 完成、结果交付、lease 释放
t6：恢复新的 TRAIN phase
```

这解释了为什么不能简单“EVAL 时拒绝所有 TRAIN start_session”：已准入 task 可能还没来得及创建 session；一刀切会让它无法完成。

但也不能只看调用方写的 `purpose=TRAIN` 就放行。必须查服务端可信的 reservation 和 phase。`purpose` 是用途，`phase_id` 是归属，`policy_version` 是模型状态；三者互不替代。

## 4. 版本兼容是重叠执行的硬条件

设旧 TRAIN session 已经用 V0 生成过一轮，EVAL 要固定为 V1。

如果系统同时要求“每条 session 从头到尾版本不变”，而 Serving 只保留一个模型版本，那么不能直接覆盖成 V1，再声称旧会话仍在 V0 上完成。

合法选择是：

- 旧 TRAIN 与 EVAL 本来就使用同一个 V，可重叠；
- 先排空旧 TRAIN，再切换为 V1；
- 明确提供多版本 Serving，并把请求路由到绑定版本。

若算法允许 TRAIN 会话跨版本，需另立语义与逐请求 provenance，不能偷偷放松约束。第一版单版本后端应显式选择兼容重叠或排空降级，而不是假设多版本能力存在。

## 5. 为什么全局 pause generation 可能造成死锁

错误顺序：

```text
全局暂停推理 → 等旧 Agent session 完成
                  ↓
旧 session 还要再调用一次模型才能完成
                  ↓
请求被暂停挡住，永远排不空
```

要停的是**新 task 准入和模型更新**，不是长时间剥夺所有已准入 session 的推理能力。

若同步权重必须 pause generation，应只在安全同步窗口等当前单次请求完成、切换权重后立即恢复；这仍不能绕过上一节的 session 版本契约。

资源上也一样：EVAL 占满 Sandbox 配额可能饿死 TRAIN tail。需要保留完成配额、优先级或直接选择非重叠方案。

## 6. 统一 Evaluation Service：三种调用方，共用一套作业语义

```text
Fixed Step Eval / Online EVO / Harness Optimization
                         ↓
                  EvaluationService
  校验与计划 → 可选 Eval Window → 调度 Episode → 执行
                         ↓
                  EvaluationCore
               评分 → 有效性 → 归约/聚合
                         ↓
                    EvalResult
```

| 组件 | 做什么 | 不做什么 |
|---|---|---|
| 上层 Controller | 为什么评、生成/选择候选、如何用结果 | 不把候选策略塞进 Gateway |
| EvaluationService | submit/status/cancel/result、作业生命周期 | 不要求部署成独立微服务 |
| PlanBuilder | 将已确定的 EVAL 请求编译为不可变计划 | 不再重复判断 TRAIN/EVAL |
| EvalWindowManager | 共享训练 Runtime 的阶段窗口 | 独立评测不必经过它 |
| EpisodeScheduler | 并发、预算、超时、重试与收口 | 不决定 fitness 公式 |
| EpisodeExecutor | 跑一条 episode、返回结果并释放资源 | 不跨 episode 聚合 |
| EvaluationCore | 评分、有效性校验、归约与指标 | 不启动 Sandbox、不调用模型、不切训练阶段 |

这里是对早期 [Evaluation Core 笔记](EvaluationCore与OnlineEVO训推链路_20260902.md) 的职责细化：编排职责放到 Service/Scheduler，Core 收窄成可独立测试的评分聚合核心。共享的是协议与语义，不强制共享所有 Runtime。

`EVAL_ROLLOUT` 是 session 用途，不是一个额外服务。Gateway 只需可信地执行 purpose/phase/version 契约，不必知道调用者是 EVO 还是 Harness。

## 7. Source 不一定只是一个数据文件

对 Agent 任务，换数据源可能同时改变任务、环境/场景路由、Reward 配置和 Runtime 要求。

因此 `source_id` 更适合引用一份有版本的 Execution Profile。上层传不透明标识，执行适配器解析细节；不要在通用 Trainer 中嵌入某个 Sandbox 的具体 Task JSON。

## 8. 最小验收矩阵

- fence 前/后并发到达的 task，只有合法旧 reservation 能继续。
- EVAL 的 success、retry、abort、timeout 结果都不会进入 TRAIN Cohort。
- EVAL 全程使用实际版本 V，不只检查请求字段写了 V。
- 单版本但 V0/V1 不兼容时，必须降级排空，不静默重绑。
- 重复 transition 幂等；旧 phase 的迟到结果归回旧 phase。
- EVAL 失败也能收口资源与版本屏障；恢复的 source cursor 不丢不重。
- 在模型请求数为零但 Agent 等待工具时，不得误报已排空。

## 9. 一分钟复习

**进 EVAL 看 fence 和 version，回 TRAIN 看执行收口；TRAIN/EVAL 隔离数据用途，不必永远隔离硬件。重叠必须满足版本兼容和资源进展条件。**

自测：task 已准入但没有 session，算不算 tail？算。模型请求数为零，能否恢复 TRAIN？不能单独据此判断。Ready cohort 能否保留？能，但消费前仍需检查版本和陈旧度。

## 整理记录

- 2026-09-07：补录 9 月 4 日的设计演进，并显式补齐单版本 Serving 的必要条件；这是设计约束澄清，不是新增实现证据。
