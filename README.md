# LLM 基础理论笔记整理

这是一个长期维护的个人 LLM 学习知识库，用来把每天的对话、实验和零散理解整理成可复习、可检索、可跨设备访问的 Markdown 笔记。

## 快速入口

- [总索引](INDEX.md)：按日期和主题查找所有正式笔记。
- [快速复习](reviews/QUICK_REVIEW.md)：下班前或开始学习前的高频复习入口。
- [每日收集箱说明](inbox/README.md)：白天如何记录尚未整理的内容。
- [每日笔记模板](templates/daily-note.md)：正式笔记的统一结构。
- [自动整理规则](AUTOMATION.md)：每日自动任务会做什么、不会做什么。
- [跨设备同步流程](SYNC_WORKFLOW.md)：在任何电脑上阅读、更新和推送前的强制同步步骤。
- [Codex 仓库规则](AGENTS.md)：其他电脑和其他 Codex 对话自动继承的约束。

## 当前学习主线

这里只保留阅读路线，逐日链接统一放在[总索引](INDEX.md)，避免首页随工作天数增长变成流水账。2026-10-10 补齐 FastVideo 历史知识，内容仍按实际发生日期归档。

- [PPO / GRPO / GSPO 复习路线](reviews/ppo-grpo-20260724-20261016.md)：统一符号与训练步时间线 → Advantage 与两层归一化 → Ratio/Clip 与 Off-policy → KL → 训推偏差 → 崩溃诊断；覆盖 7 月 24 日～10 月 16 日。
- [10 月近期学习路线](reviews/QUICK_REVIEW.md)：并行切分、路由显存、权重版本、Agentic RL 吞吐与健康负载对照。
- [FastVideo 与 WM 复习路线](reviews/fastvideo-wm-20260918-20261007.md)：视频扩散 → 条件广播 → 编译边界 → BF16 梯度 → 显存生命周期 → 长跑证据；覆盖 9 月 18 日～10 月 7 日补录。
- [9 月 W38 复习路线](reviews/2026-W38.md)：输出端梯度 → 编译布局 → 评测终态 → 显存与保存，保留为前置阅读。

## 按月浏览

| 月份 | 主题概览 | 月度入口 |
|---|---|---|
| 2026 年 10 月 | PPO/GRPO/GSPO、KL 与训推偏差、RL 崩溃归因；PP/CP/SP、路由回放与动态形状、权重版本、Agent RL 吞吐与健康窗口 | [查看 10 月全部笔记](INDEX.md#2026-年-10-月) |
| 2026 年 9 月 | MTP/评测基础；FastVideo 视频扩散、编译融合、BF16 验收与显存生命周期 | [查看 9 月全部笔记](INDEX.md#2026-年-9-月) |
| 2026 年 8 月 | Online EVO Rollout、MTP 显存生命周期与 allocator 归因、Online MTP K/D 解耦、MTP+CP/PP、Cohort、MoE Router Replay | [查看 8 月全部笔记](INDEX.md#2026-年-8-月) |
| 2026 年 7 月 | LLM/Transformer 基础、训练与 RL、MoE/MTP、状态管理、Qwen3.5 GDN | [查看 7 月全部笔记](INDEX.md#2026-年-7-月) |

## 推荐使用方式

### 打开仓库后：先同步

无论准备阅读、更新还是让 Codex 总结仓库，都先运行：

```bash
./scripts/sync-before-work.sh
```

推送前，在本地提交完成且工作区干净后，再运行一次同步脚本，然后才能：

```bash
git push origin main
```

新电脑首次 clone 后运行：

```bash
./scripts/bootstrap-repo.sh
```

### 白天：只管捕获

学习过程中，不要求马上把内容写得很完整。将原始材料追加到：

```text
inbox/YYYY-MM-DD.md
```

可以记录：

- 今天和 Codex 讨论的问题；
- 自己尚未确定的理解；
- 命令、日志和实验现象；
- 想在下班前整理的问题；
- 值得长期保留的结论。

在 Codex 中也可以直接说：

```text
把刚才这段学习内容追加到 llm-foundations-notes 当天的 inbox。
```

Codex 追加时应记录北京时间、内容归属日期和“未整理”状态。18:25 之后学到的内容仍然写入当天 Inbox。

### 每天 18:25：主整理

每天 18:25 的自动任务会：

1. 同步 GitHub 最新状态；
2. 枚举 Codex “2026 Task”项目中的全部对话；
3. 必查两个已登记的基础理论对话，并增量深读其他新增或变化的对话；
4. 只提炼可复用的基础理论，排除公司工作安排和敏感上下文；
5. 阅读总索引、快速复习页、最近几篇正式笔记和当天 `inbox`；
6. 去重、纠错并补齐上下文；
7. 生成当天正式笔记；
8. 更新总索引和复习队列；
9. 运行基础检查；
10. 提交并推送到 GitHub。

如果当天没有实质性 `inbox` 内容，任务不会凭空生成学习记录，也不会创建无意义提交。

### 次日 09:25：只补漏昨晚内容

次日 09:25 的任务会检查前一天 18:25～23:59 新增、但尚未进入正式笔记的内容：

- 只处理“未整理”的内容；
- 再和前一天正式笔记比对，防止重复；
- 仍然写回前一天的正式笔记；
- 在正式笔记中记录实际补录时间；
- 成功后把 Inbox 条目标记为“已补录”；
- 没有缺失内容时不创建提交。

### 第二天：先复习再学习

建议开始新内容前依次阅读：

```text
reviews/QUICK_REVIEW.md
→ 前一天正式笔记的“一分钟总结”
→ 前一天正式笔记的“自测问题”
→ 当天新内容
```

## 仓库结构

```text
llm-foundations-notes/
├── README.md                    # 使用说明和入口
├── AGENTS.md                    # Codex 跨设备强制规则
├── INDEX.md                     # 日期索引、主题索引
├── AUTOMATION.md                # 自动整理规范
├── SYNC_WORKFLOW.md             # 跨设备同步流程
├── .githooks/
│   └── pre-push                 # 推送前远端新鲜度检查
├── scripts/
│   ├── bootstrap-repo.sh        # 新电脑初始化
│   └── sync-before-work.sh      # 阅读、更新、推送前同步
├── inbox/                       # 当天未整理的原始记录
│   └── README.md
├── notes/                       # 已整理的正式笔记
│   └── YYYY/MM/
├── reviews/
│   └── QUICK_REVIEW.md          # 快速复习与间隔复习队列
└── templates/
    └── daily-note.md            # 每日正式笔记模板
```

## 核心原则

- 原始记录和正式笔记分离；
- 正式笔记必须有日期、主题和复习摘要；
- 不重复堆砌旧内容，而是建立链接；
- 不确定内容明确标记，不把推测写成事实；
- 一次实验尽量只改变一个主要变量；
- GitHub 中的内容以 Markdown 为主，保证手机和电脑都能快速阅读；
- 每次自动提交只包含本知识库范围内的文件。
- GitHub `origin/main` 是跨设备共享事实来源；
- 每次阅读、更新前先执行 fast-forward-only 同步；
- 每次推送前提交本地修改、再次同步，再进行推送；
- 禁止自动 stash、reset、重写历史或 force push。
