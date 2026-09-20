# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

**[wiki/index.md](wiki/index.md) — read it before grepping the codebase.** `wiki/` is the compiled knowledge base for this repo; the index locates the right page in one read. See [The wiki](#the-wiki--read-it-first-keep-it-current) below.

[AGENTS.md](AGENTS.md) is the canonical agent guide for this codebase (setup, commands, registry/agent/domain patterns, code style, commit conventions). Two scoped companions cover their directories in more detail:

- [src/tau3/domains/AGENTS.md](src/tau3/domains/AGENTS.md) — required files per domain, `ToolType` semantics, split-file rules
- [tests/AGENTS.md](tests/AGENTS.md) — test tiers and fixtures

This file covers only what is specific to **this fork** and where the upstream docs have drifted from the code.

## The wiki — read it first, keep it current

`wiki/` is a knowledge base compiled over this repository and maintained entirely by the agent, following the [llm-wiki](wiki/llm-wiki.md) pattern. It exists because this repo's own docs drift behind the code and the load-bearing knowledge — evaluation mechanics, retrieval defaults, known environment defects — is spread across many files. It is **gitignored**: local to this checkout, never committed.

**Start every session that touches this repo's behavior at [wiki/index.md](wiki/index.md).** It is a one-page catalog — locate the page, drill in, and read source files only when no page covers the question or the page is marked `status: stale`. [wiki/overview.md](wiki/overview.md) orients a cold start (what this fork is, reading order, the three traps); [wiki/log.md](wiki/log.md) says what happened recently.

The wiki is only as good as its last update, so close the loop:

| When | Do |
|---|---|
| You answered a question by real synthesis — a comparison, a mechanism walk-through, a failure analysis | File it back as a page under `wiki/notes/`, then update `wiki/index.md` and `wiki/log.md`. Otherwise the work is lost to chat history. |
| You changed code or docs that a page cites | Update those pages ([wiki/sources.md](wiki/sources.md) maps source → pages), refresh their `sources:` (`path@commit`) and `updated:` frontmatter, append to `wiki/log.md`. |
| After a large merge, or ~every 10 ingests | Run the drift / orphan / dead-link / filename-uniqueness lint in [wiki/CLAUDE.md](wiki/CLAUDE.md). |

[wiki/CLAUDE.md](wiki/CLAUDE.md) is the schema — **read it before writing anything under `wiki/`**; inside that directory it takes precedence over this file. Its standing rules: truth precedence is **code > docs > inference**, and a conflict is recorded in `wiki/notes/contradictions.md` rather than silently resolved; pages are written in **English** even when the conversation is Chinese; wikilinks use the bare filename (`[[evaluation-and-reward]]`, never a path).

## This fork

Branch `gym/tau3`, forked from upstream `tau2-bench`. Three things distinguish it from upstream:

1. **Package renamed `tau2` → `tau3`** (commit `efb0a80`); the CLI entry point is `tau3`.
2. **Domains trimmed** (commit `363b67b`): `airline`, `retail`, `telecom` were removed. `src/tau3/domains/{airline,retail,telecom}/` and `tests/test_domains/test_{airline,retail,telecom}/` still exist as **empty directories** — AGENTS.md and tests/AGENTS.md still list these as working domains/test tiers. Registered domains are only `banking_knowledge` and `mock`.
3. **Task synthesis pipeline** (`src/tau3/synthesis/`, uncommitted) — the main body of active work, plus two unregistered domains (`hotel`, `movie`).

## Task synthesis pipeline

`src/tau3/synthesis/` generates new training tasks against a **fixed** `banking_knowledge` world: knowledge base and business tools stay untouched; customers, account state, goals and reference actions are synthesized. Full spec: [docs/task-synthesis.md](docs/task-synthesis.md). Config: [configs/synthesis/tau3-aa.yaml](configs/synthesis/tau3-aa.yaml).

Stages (`tau3 synthesize <stage>`, wired in [src/tau3/synthesis/cli.py](src/tau3/synthesis/cli.py) and attached to the main parser in [cli.py:414](src/tau3/cli.py#L414)):

```
build-catalog → generate → validate → export → collect-sft
                                   ↘ repair-bundle (fork a draft bundle)
```

Module map: `workflow.py` drives the stages; `scenarios.py` builds candidate scenarios per family (selection / cashback / credit-limit / ordering); `validation.py` does schema checks, independent business-result checks, strict real-tool replay and deterministic DB comparison; `llm.py` wraps the four LLM roles (generator / teacher / user / judge, all `openai/GLM5.3-agentic-qs-h20` via LiteLLM's OpenAI-compatible adapter); `bundle.py` + `storage.py` handle the on-disk bundle format.

Bundle invariants worth knowing before touching this code:
- Bundles are resumable: `candidates/<slot>.json` checkpoints are written atomically, and `--resume` requires an identical config and task count.
- `manifest.json` flips from `draft` to `published` only after admission. `load_task_bundle` rejects draft, altered, incompatible or malformed bundles.
- `--offline` produces deterministic dev text. Offline candidates must never be published or cited as evidence of a live conversation.
- Concurrent writers to one bundle directory are unsupported.
- `scripts/audit_synthesis_bundle.py` is a separate read-only delivery audit — run it before claiming a bundle is accepted.
- `scripts/run_synthesis_batch.py` is the unattended pilot→200-task→SFT wrapper.

### `--task-bundle` plumbing

A published bundle can substitute for registry task loading. The path threads through:

`cli.py` `--task-bundle` → `RunConfig.task_bundle` → [runner/helpers.py:68](src/tau3/runner/helpers.py#L68) `get_tasks()` → `load_task_bundle()`, and separately into the gym via `AgentGymEnv(task_bundle=..., task_split_name=...)`.

[gym/gym_agent.py](src/tau3/gym/gym_agent.py) `_bundle_retrieval()` resolves the retrieval config from the bundle manifest **only when not explicitly passed** — explicit `--retrieval-config` stays an intentional diagnostic override. Preserve that precedence when editing. Registry loading remains the path when no bundle is given.

## Unregistered domains: hotel and movie

Both live under `src/tau3/domains/` with data in `data/tau3/domains/`, and neither is registered in [registry.py](src/tau3/registry.py) — so **neither is reachable from the CLI by default**.

- `hotel` self-registers through `register()` in [domains/hotel/environment.py](src/tau3/domains/hotel/environment.py), which must be called explicitly. That call also runs `dialogue_scoring.install()`, which **monkeypatches `NLAssertionsEvaluator.calculate_reward`** to route `hotel` tasks to a custom audit-based reward while delegating every other domain to the original. Be aware of this global patch when debugging evaluator behavior. Hotel is interactive-text-only (`solo_mode` raises).
- `movie` has no `register()` at all; it is wired up ad hoc. Evaluation notes: [data/tau3_movie_评测纪要.md](data/tau3_movie_评测纪要.md).

Hotel code and data are written in Chinese (policy, tool docstrings, error messages) and use a dense, compact style unlike the rest of the repo — match the surrounding file rather than the repo default when editing there.

## Testing this fork

`make test` (core) and `make test-gym` work as documented. `make test-knowledge` now also runs the synthesis suites — `test_synthesis.py`, `test_synthesis_coverage.py`, `test_synthesis_review_budget.py`, `test_synthesis_sft_budget.py` in `tests/test_domains/test_banking_knowledge/` — so synthesis changes need `uv sync --extra knowledge --extra dev`.

Single file / single test:

```bash
uv run pytest tests/test_domains/test_banking_knowledge/test_synthesis.py
uv run pytest tests/test_domains/test_banking_knowledge/test_synthesis.py -k budget
```

Run `make check-all` (ruff lint + format) before committing; a pre-commit hook enforces it.

## Corrections to AGENTS.md

- The default `--retrieval-config` for `banking_knowledge` is **`bm25_grep`** (fully offline, no API key), per [cli.py:231](src/tau3/cli.py#L231). AGENTS.md's "Things to Watch Out For" section says `alltools`; that is stale and would imply a needless `OPENAI_API_KEY` + `sandbox-runtime` dependency.
- `make test` is described as covering "airline, retail, telecom, mock" — those three domains and their test directories are now empty.
