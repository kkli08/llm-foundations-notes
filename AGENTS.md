# Repository Instructions

These instructions apply to the entire `llm-foundations-notes` repository on every computer and in every Codex task.

## Mandatory remote sync gate

Except for reading this `AGENTS.md` to learn the rules, do not read, summarize, edit, generate, commit, or push repository content until the local checkout has been synchronized with `origin/main`.

At the start of every task:

```bash
./scripts/sync-before-work.sh
```

The script requires:

- the current branch to be `main`;
- a clean working tree;
- a configured `origin`;
- a successful `git pull --ff-only origin main`.

If synchronization fails because of uncommitted files, network access, authentication, or divergent branches:

- stop before reading or editing repository content;
- report the exact category of blocker;
- preserve all local and remote work;
- never automatically stash, reset, discard, rebase, merge, or force-push.

Minimal read-only diagnostics such as `git status -sb`, `git remote -v`, and `git log --oneline --decorate -n 5` are allowed to identify the blocker.

## Before every update

An “update” includes adding an Inbox entry, editing a note, updating an index, changing automation documentation, or modifying repository tooling.

Required order:

1. Run `./scripts/sync-before-work.sh`.
2. Read the latest relevant files.
3. Make the scoped change.
4. Validate Markdown, links, dates, and sensitive information.
5. Commit only the intended files.

Do not keep uncommitted Inbox changes across computers. When Codex captures a new Inbox entry, it should sync first, append the timestamped entry, commit it, sync again, and push it so another computer can see it.

## Before every push

After committing and while the working tree is clean:

```bash
./scripts/sync-before-work.sh
git push origin main
```

The second synchronization is mandatory even if the repository was synchronized at task start. If another computer pushed meanwhile, `git pull --ff-only` will fail instead of silently creating a merge or overwriting history.

The versioned pre-push hook is an additional safety net. It fetches `origin/main` and rejects a push when the local branch does not contain the latest remote commit. It does not replace the mandatory pre-push synchronization.

Never use `git push --force` or `git push --force-with-lease` in this repository.

## New computer bootstrap

After cloning the repository on another computer, run:

```bash
./scripts/bootstrap-repo.sh
```

This configures the repository-local hooks and fast-forward-only pull behavior, then performs the initial synchronization.

Codex should run the bootstrap script when it detects that `core.hooksPath` is not `.githooks`.

## Knowledge workflow

After synchronization, follow:

- `README.md` for the knowledge-base structure;
- `SYNC_WORKFLOW.md` for cross-device synchronization;
- `AUTOMATION.md` for the 18:25 主整理 and 09:25 次日补漏 processes;
- `inbox/README.md` for timestamp and processing-state conventions.

Repository files are the shared source of truth. Local Codex automation registrations may be machine-specific, so another computer must read `AUTOMATION.md` and recreate those scheduled tasks there only if that computer should run them.
