# Contributing to Asshole AI Detector

We work through **GitHub Issues → feature branch → Pull Request → review → merge**. Never push feature work directly to `main`.

## 1. Setup

```bash
git clone <repository-url>
cd asshole-ai-detector
uv sync
uv run pytest
uv run ruff check .
```

Use the project interpreter from `.venv` (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux/macOS). Python version is pinned in `.python-version`.

## 2. Pick up a task

1. Open the GitHub Project board and pick an unassigned, unblocked Issue.
2. Assign yourself and move it from **Todo** to **In Progress**.
3. Read its description, acceptance criteria, and blocking dependencies.
4. Discuss changes to shared Pydantic schemas or architecture before coding.

## 3. Branch, implement, test, and PR

Example for Issue **#7**:

```bash
git switch main
git pull origin main
git switch -c feat/consensus-labeling

# Implement changes and tests
uv run pytest
uv run ruff check .

git add .
git commit -m "feat(data): implement consensus labeling"
git push -u origin feat/consensus-labeling
```

Open a PR with **base `main`**, include `Closes #7` in its description, and request a teammate review. Move the issue to **In Review**. Merge only when checks pass and review is complete. After merge:

```bash
git switch main
git pull origin main
git branch -d feat/consensus-labeling
```

Move the issue to **Done** (or allow the configured board automation to do so).

## 4. Names and expectations

- Branches: `feat/data-preprocessing`, `fix/consensus-labels`, `docs/annotation-guidelines`.
- Commits: `feat(data): ...`, `fix(rag): ...`, `test(data): ...`, `docs: ...`, `chore: ...`.
- PRs: one focused change; description, related issue, test results, and important limitations.
- Tests: include unit tests for data rules, split integrity, public schemas and control-flow decisions. Mock external models when possible.
- Reproducibility: record seeds, model revisions, configs and experiment IDs; keep Golden data out of tuning.
- Privacy/security: **never commit `.env`, tokens, original datasets, weights, Chroma indexes, or user-submitted stories**. Commit changes to `pyproject.toml` and `uv.lock` together when dependencies change.

## 5. Review checklist

- [ ] Implements issue acceptance criteria, with no unrelated changes.
- [ ] Matches `docs/architecture.md` and shared Pydantic contracts.
- [ ] Relevant tests added or a reason provided.
- [ ] `uv run pytest` and `uv run ruff check .` pass.
- [ ] No generated artifacts, private data, or credentials included.
- [ ] Reviewer approval obtained before merge.

Current project phases and research requirements are in `docs/asshole-ai-detector-milestone-2.md`.
