#!/bin/sh

set -eu

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "ERROR: 当前目录不在 Git 仓库中。" >&2
  exit 1
}

cd "$repo_root"

branch=$(git branch --show-current)
if [ "$branch" != "main" ]; then
  echo "ERROR: 当前分支是 '$branch'，要求在 main 上同步。" >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "ERROR: 未配置 origin，无法确认远端最新状态。" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: 存在已跟踪但未提交的修改。为避免覆盖，未执行 pull。" >&2
  git status -sb >&2
  exit 1
fi

untracked=$(git ls-files --others --exclude-standard)
if [ -n "$untracked" ]; then
  echo "ERROR: 存在未跟踪文件。请先确认、提交或移出仓库，再同步。" >&2
  printf '%s\n' "$untracked" >&2
  exit 1
fi

echo "Syncing origin/main with fast-forward-only policy..."
git pull --ff-only origin main

echo "Repository is synchronized and clean."
git status -sb
