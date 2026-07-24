#!/bin/sh

set -eu

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "ERROR: 当前目录不在 Git 仓库中。" >&2
  exit 1
}

cd "$repo_root"

git config --local core.hooksPath .githooks
git config --local pull.ff only
git config --local fetch.prune true

echo "Configured repository-local Git settings:"
echo "  core.hooksPath=$(git config --local --get core.hooksPath)"
echo "  pull.ff=$(git config --local --get pull.ff)"
echo "  fetch.prune=$(git config --local --get fetch.prune)"

./scripts/sync-before-work.sh
