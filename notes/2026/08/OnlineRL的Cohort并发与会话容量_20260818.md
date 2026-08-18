# Online RL 的 Cohort 并发与会话容量

> 日期：2026-08-18
> 来源：Codex Session `019ff0f2-6a27-7220-9da5-532f3f08bea2`
> 状态：已整理

## 0. 这篇笔记最终要解决的问题

在 GRPO/Online RL 中，同一 Prompt 要生成多条回答。为什么集群看起来有很多 Rollout Backend，仍可能因为单后端 Session 容量不足而超时？全局并发、每后端物理容量、Cohort 大小和 Reservation Timeout 应怎样配合？

## 1. 先给结论

1. Cohort 是同一 Prompt 的一组 Response/Session；Group-relative Advantage 通常要等整组结果就绪后才能计算。
2. `global concurrency` 会被分配到多个 Rollout Backend。若一个 Cohort 固定落到同一 Backend，该 Backend 的物理 Session 容量至少要能容纳整个 Cohort。
3. 增大 Reservation Timeout 只会允许排队更久，不能修复物理容量不足或串行生成导致的吞吐问题。
4. 预约/容量失败是首因时，Session 被清理后出现的鉴权失败通常是次生错误；必须按时间线抓第一条因果错误。
5. Judge/Reward 服务是 Online RL 的外部依赖，必须通过模型列表和真实请求验证，不能只看端口打开或进程存活。
6. MTP Off/On A/B 必须固定 Cohort、并发、后端数、生成上限、Judge、数据和 Seed；否则测到的可能是调度差异而非 MTP。

## 2. 五个容易混淆的对象

| 对象 | 含义 |
|---|---|
| Prompt | 一道题或一次输入 |
| Response / Session | 针对 Prompt 的一条独立生成轨迹 |
| Cohort / Group | 同一 Prompt 的多条 Response 集合 |
| Rollout Backend | 实际执行 vLLM/SGLang 生成的服务实例 |
| Reservation | Workflow 为一组 Session 申请执行容量的过程 |

例如 `n_samples=8`：

```text
Prompt P
├── Session 1 → Response 1
├── Session 2 → Response 2
...
└── Session 8 → Response 8

这 8 条共同组成一个 Cohort
```

GRPO 的组内 Mean/Std 或排序依赖整组 Reward，因此一条完成并不代表该 Prompt 已可进入 Actor Update。

## 3. Global Concurrency 与 Physical Session Capacity

假设：

- 全局最大并发为 $C$；
- Rollout Backend 数为 $R$；
- Cohort 大小为 $G$。

若容量近似平均切分，每个 Backend 的物理容量约为：

$$
C_{backend}\approx\left\lfloor\frac{C}{R}\right\rfloor
$$

若一个 Cohort 被固定到同一 Backend，最低条件是：

$$
C_{backend}\ge G
$$

### 一个直观例子

```text
8 个 Rollout Backends
Cohort Size = 8
```

若 Global Concurrency = 8：

```text
每 Backend 容量约 1
→ 一个需要 8 个 Session 的 Cohort 只能串行或无法整体预约
```

若 Global Concurrency = 64：

```text
每 Backend 容量约 8
→ 每个 Backend 可以同时承载一个完整 Cohort
```

这不是说 64 永远正确，而是说明容量计算必须看“每后端能否容纳业务调度单元”，不能只看集群总并发。

## 4. 为什么 Cohort 可能固定到一个 Backend

把同一 Cohort 固定到一个 Backend 可能为了：

- Session 生命周期和状态一致；
- Prefix/Tokenizer/模型版本一致；
- 简化 Reward 回调和清理；
- 避免一组请求跨服务产生额外协调。

但代价是全局空闲容量不能总被这一 Cohort 使用。即使其他 Backend 很空，只要目标 Backend 没有足够 Slot，该 Cohort 仍会排队。

所以调度分析要同时记录：

- Global Concurrency；
- Backend 数量；
- 每 Backend 实际 Capacity；
- Cohort Affinity/分片规则；
- 每个 Cohort 的 Session 数。

## 5. Reservation Timeout 能解决什么，不能解决什么

Reservation Timeout 表示 Cohort 等待容量的最长时间。

增大 Timeout 可以解决：

- 前一批长回答尚未释放 Slot；
- 正常长尾导致短时间排队；
- 服务冷启动后第一批 Capacity 暂未就绪。

增大 Timeout 不能解决：

- 每 Backend 容量永远小于 Cohort Size；
- 长回答完全串行，吞吐根本不足；
- Session 泄漏不释放；
- Backend 路由或 Affinity 配置错误。

因此正确顺序是：

```text
先证明 Capacity/分片能容纳 Cohort
→ 再根据正常长尾设置 Timeout
```

不能用一个很大的 Timeout 掩盖容量设计错误。

## 6. 从首因到次生错误的时间线

典型失败链可能是：

```text
Cohort 等待容量超过 Reservation Deadline
→ Start Session 被拒绝
→ Workflow 清理/回收该 Session
→ 迟到的 Reward 或 Callback 继续访问
→ Session Key 已失效
→ 出现 Unauthorized / Invalid Session
```

如果只看最后一条鉴权错误，可能误判为 Secret、Judge 或认证系统问题。正确归因应：

1. 按时间排序所有相关日志；
2. 找第一条改变系统状态的错误；
3. 区分 Primary Failure 与 Cleanup Consequence；
4. 修首因后重新观察次生错误是否自然消失。

这也是“第一因果错误优先”在分布式 Online RL 中的具体应用。

## 7. Long-tail 为什么对 Group Rollout 更敏感

一组 Response 的完成时间近似由最慢样本决定：

$$
T_{cohort}\approx\max(T_1,T_2,\ldots,T_G)
$$

长回答、工具调用、Judge 延迟都会拖住整组 Advantage 计算。即使平均生成很快，P95/P99 长尾仍可能造成：

- Cohort 长时间占用 Session Slot；
- 后续 Cohort 无法预约；
- Actor 等待 Rollout，产生 Pipeline Bubble；
- Timeout 和清理错误增加。

所以 Online RL 不能只看平均 Tokens/s，还要看：

- Cohort Ready Latency；
- Session Occupancy；
- Reservation Wait p50/p95；
- Response Length Distribution；
- Judge/Reward Tail Latency。

## 8. Judge 服务为何是独立正确性门

一个 Online Math/RL Cohort 通常还依赖 Judge：

```text
Rollout Response
→ Judge / Verifier
→ Reward
→ Group Advantage
→ Actor Update
```

Judge 的最低 Readiness 不能只是“Job Running”或“端口能连接”，而应包括：

1. 模型列表/健康接口成功；
2. 使用真实格式发一次请求；
3. 返回结构、Tokenizer/Template 和 Score 语义符合预期；
4. 从训练 Job 所在网络实际可达；
5. 生命周期覆盖整场实验。

如果 Judge 独立部署，还应先 Ready，再启动昂贵训练任务；实验结束或放弃后及时释放。

## 9. 配置组合：请求值必须真正进入运行时

Hydra 等配置系统通常区分：

```text
覆盖已有键：key=value
新增原配置中不存在的键：+key=value
```

若把新增键当普通覆盖，任务可能在配置解析阶段就失败；反过来，对已有键乱加 `+` 也可能产生重复或拒绝。

提交前应对原始 YAML 做静态检查：

```text
完整 Key 已存在？
├── 是：普通 Override
└── 否：显式 Add Override
```

然后在运行时 Preflight 打印最终组合后的关键值。命令行写了什么仍不是最终事实，必须形成：

```text
Requested → Composed → Runtime Selected → Measured
```

## 10. MTP Off/On 严格 A/B 如何固定变量

为了比较 MTP，至少固定：

- 同一模型与权重版本；
- 数据、Prompt、Seed；
- Cohort Size / `n_samples`；
- Global Concurrency 和 Backend 数；
- Max Tokens / Context Length；
- Sampling；
- Judge/Reward；
- Actor/Rollout 并行拓扑；
- CUDA Graph、Attention Backend、Profiler；
- Warmup 和统计 Step。

推荐顺序：

```text
先完成 MTP-Off 全流程
→ 确认资源释放和 Judge 仍 Ready
→ 再启动完全匹配的 MTP-On
```

这样可以减少资源竞争、Judge 状态和并发重叠对 A/B 的污染。

短实验只能回答系统链路和早期性能，不能证明最终模型质量。Reward 也只是训练样本表现，独立 Held-out Math Eval 才回答能力变化。

## 11. 观测指标分层

### 11.1 调度与容量

- Active/Pending Sessions；
- Physical Capacity per Backend；
- Reservation Wait；
- Cohort Ready/Timeout Count；
- Backend Queue Length。

### 11.2 Rollout

- Output Tokens/s；
- TTFT/TPOT；
- Response Length p50/p95；
- MTP Acceptance Length/Rate；
- 每 Cohort 最慢 Session 时间。

### 11.3 Judge/Reward

- Judge Request p50/p95；
- Success/Error Count；
- Reward Mean/Distribution；
- Cohort 内有效 Reward 数。

### 11.4 Actor Update

- Actor/MTP Loss；
- Grad Norm、NaN/Inf；
- KL、Clip Ratio；
- Optimizer Success；
- Weight Version Sync 和下一轮消费版本。

## 12. 最小验证阶梯

1. 配置 Compose：检查新增/覆盖键和最终运行值；
2. 数据 Preflight：Dataset Schema、样本数量、Prompt 格式；
3. Judge Preflight：跨 Job/节点真实请求；
4. Rollout Capacity Smoke：一个完整 Cohort 能同时 Ready；
5. 两步闭环：两轮 Rollout/Reward/Update/Weight Sync；
6. Matched Off/On：固定所有非 MTP 变量；
7. 长跑/Eval：再判断稳定性和模型质量。

## 13. 常见误解

| 容易误解为 | 更准确的理解 |
|---|---|
| 集群 Global Concurrency 足够就不会排队 | 还要看每 Backend Capacity 和 Cohort Affinity |
| Timeout 调大就修好了 | 它只延长等待，不能增加吞吐或物理 Slot |
| 最后一条 Unauthorized 是首因 | 可能是 Reservation 失败清理后的次生错误 |
| 一个 Session 完成就能计算 GRPO Advantage | 通常要等同 Prompt 的完整 Cohort |
| Judge Job Running 就算 Ready | 还要做真实请求和跨网络可达性验证 |
| 两步 Reward 上升就是模型能力提升 | 训练 Reward 不是 Held-out Eval，短跑不证明质量 |

## 14. 一分钟复习

1. Cohort 是同 Prompt 的多条 Session；Group Advantage 往往等整组完成。
2. 若 Cohort 固定在一个 Backend，要求 `per-backend capacity >= cohort size`。
3. Timeout 只能容忍正常排队，不能修复容量不足。
4. Reservation 失败后出现的鉴权错误可能是次生错误，要抓第一因果错误。
5. MTP A/B 必须固定 Cohort、并发、后端、Judge、数据和生成长度。

## 15. 自测问题

### 问题 1

8 个 Backend、Global Concurrency=8、Cohort Size=8，为什么仍可能无法并发启动一个 Cohort？

期望回答：容量若平均分片，每个 Backend 只有 1 个 Slot；而 Cohort 固定在一个 Backend 时需要 8 个 Slot。

### 问题 2

为什么把 Timeout 从 30 分钟改成 90 分钟不一定解决问题？

期望回答：如果根因是每后端容量不足或生成串行，延长 Timeout 只让请求等更久，没有增加吞吐和可用 Slot。

### 问题 3

为什么先看到 Reservation Error，随后看到 Unauthorized 时，应优先查前者？

期望回答：Reservation 失败可能触发 Session 清理；后续迟到请求使用已失效 Session Key，Unauthorized 是结果而非根因。

## 16. 与已有知识的联系

- [On-policy 训推与 MTP 性能分析](OnPolicy训推与MTP性能分析_20260806.md)：本文补充 Group Rollout 的排队、容量和长尾。
- [训练部署 Baseline 与 RLVR 闭环](../07/训练部署Baseline与RLVR闭环_20260727.md)：将“平台 Running 不等于业务成功”推进到 Session/Cohort 层。
- [推理性能的配置执行证据链](推理性能的配置执行证据链_20260812.md)：补充 Hydra Compose 和 Runtime Preflight。

## 17. 尚未解决与后续路线

- Cohort 跨 Backend 分散调度与 Affinity 的权衡。
- Continuous Batching 下 Session Capacity 与 KV Cache 容量的联合模型。
- Reward/Judge 长尾对 Actor-Rollout 异步流水线的影响。
- 长跑中动态并发、Backpressure 和 Admission Control。
