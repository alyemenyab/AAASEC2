---
name: commit-message
description: Write a Conventional Commits message from a description of a change, a diff, or a list of edited files. Use when asked for a commit message, to phrase a commit, or to clean up commit history wording.
---

# Commit Message

Produce exactly one commit message. Nothing before it, nothing after it.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type** — one of: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`.
- **scope** — the smallest accurate area of the codebase (`api`, `mcp`, `agent`, `compose`). Omit the parentheses entirely if the change is repo-wide.
- **subject** — imperative mood, lower case, no trailing period, **50 characters or fewer**. "add health endpoint", never "added" or "adds".
- **body** — optional; include it only when the change is not self-evident. Wrap at 72 characters. Explain **why**, not what — the diff already says what.
- **footer** — optional; `BREAKING CHANGE: <consequence>` or `Refs #123`.

## Procedure

1. Identify the single dominant change. If the input describes two unrelated changes, say so in one line and write **two** messages — do not blend them into one.
2. Pick the type from the effect on the user or the API, not from the file extension. A change to a test file that fixes a real bug is `fix`, not `test`.
3. Write the subject first, then delete every word it does not need.
4. Add a body only if a reviewer six months from now would ask "why?".

## Rules

- Never write `update`, `changes`, `stuff`, `various`, `misc`, or `WIP` as the subject.
- Never mention the tool that produced the change, the author, or the date.
- Never exceed 50 characters in the subject line. Count them.
- Output the raw message only — no code fence, no commentary, no alternatives.

## Example

Input: "I made the FastAPI service build the deep agent once when the module loads instead of on every request, because startup was being repeated per call."

Output:

```
perf(api): build agent once at import time

Constructing the agent per request re-created the model client and
re-read the skills directory on every call, adding latency to each
response for no benefit. The agent is stateless between requests,
so one module-level instance is enough.
```
