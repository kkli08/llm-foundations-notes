# 跨设备同步流程

## 目标

无论在当前 Mac、另一台电脑还是另一个 Codex 对话中工作，都遵循：

```text
先同步远端
→ 再阅读
→ 再更新
→ 提交
→ 推送前再次同步
→ 推送
```

GitHub 的 `origin/main` 是共享事实来源。任何设备都不能假设自己的本地文件已经是最新版本。

## 第一次在新电脑使用

```bash
gh repo clone kkli08/llm-foundations-notes
cd llm-foundations-notes
./scripts/bootstrap-repo.sh
```

Bootstrap 会为当前 clone 设置：

```text
core.hooksPath = .githooks
pull.ff = only
fetch.prune = true
```

这些是仓库本地 Git 配置，不会污染其他仓库。

## 每次阅读前

先运行：

```bash
./scripts/sync-before-work.sh
```

只有出现下面结果后，才开始阅读笔记：

```text
Already up to date.
或
成功 fast-forward 到 origin/main
```

如果同步失败，不要基于可能过期的本地笔记做总结或计划。

## 每次更新前

```bash
./scripts/sync-before-work.sh
```

然后再编辑文件。

对于白天新增的一条 Inbox：

```text
同步
→ 写入带北京时间和“未整理”状态的条目
→ 提交
→ 再次同步
→ 推送
```

推荐提交信息：

```text
inbox: 记录 2026-07-24 21:10 学习内容
```

这样另一台电脑和晚间自动任务都能看到这条内容。

## 每次推送前

必须先把本地修改提交，确保工作区干净：

```bash
git status -sb
./scripts/sync-before-work.sh
git push origin main
```

如果另一台电脑已经推送了新提交，本地 `main` 与远端会产生分叉，`git pull --ff-only` 将拒绝继续。

此时：

- 不要强推；
- 不要自动 reset；
- 不要自动 rebase 或 merge；
- 先检查两边提交内容，再让用户决定如何整合。

## Pre-push Hook 的作用

`.githooks/pre-push` 会在 Git 真正推送前：

1. 确认当前分支是 `main`；
2. 确认工作区干净；
3. Fetch 最新 `origin/main`；
4. 确认本地 HEAD 已包含最新远端提交；
5. 如果不满足则拒绝推送。

Hook 是最后一道防线，不代替主动运行：

```bash
./scripts/sync-before-work.sh
```

## 未提交修改如何处理

同步脚本发现未提交修改时会停止。

这是有意设计：

- 不自动 stash，避免内容在另一台电脑不可见；
- 不自动 commit，避免提交范围不明确；
- 不自动 reset，避免丢失学习记录；
- 不在脏工作区中 pull，避免覆盖或冲突。

正确做法是先确认这些修改属于什么：

- 有价值的 Inbox：检查后提交并同步；
- 正在编辑的正式笔记：完成、验证并提交；
- 不确定或冲突内容：停止并由用户决定；
- 敏感信息：不要提交，先移除。

## Codex 跨设备继承

根目录 `AGENTS.md` 是 Codex 的仓库级工作规则。将仓库 clone 到其他电脑并在 Codex 中打开后，Codex 应先读取该文件，从而知道：

- 阅读前先同步；
- 更新前先同步；
- 推送前再次同步；
- 禁止自动丢弃、stash、重写或强推；
- 自动化日程与知识归档规则在哪里。

注意：

> GitHub 会同步仓库文件和流程说明，但 Codex 的本地定时任务注册可能属于具体电脑。需要在哪台电脑运行 18:25 和 09:25 自动任务，就在那台电脑根据 `AUTOMATION.md` 检查或创建任务；避免多台电脑同时运行造成重复提交。
