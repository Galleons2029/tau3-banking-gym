# Domain perturbation: variant domains for overfitting control

## Why

Task synthesis runs against one official `banking_knowledge` corpus. A model can
score well on it without learning to retrieve and reason — by memorising surface
symbols instead: the brand `Rho-Bank`, product names like `Light Green Account`,
tool names like `submit_cash_back_dispute_0589`, and document ids such as
`doc_checking_accounts_light_green_account_002`. Synthesising more tasks does not
help, because they all sit on the same symbol set.

This package derives **variant domains**: same business structure, different
symbols. Train on one set of variants, evaluate on a held-out variant and on the
untouched official domain, and the score gap measures the memorisation directly.

## What is guaranteed

- **The canonical domain is never modified.** Input comes from a frozen snapshot;
  output goes to `data/variants/`. `environment_fingerprint()` is unchanged, so
  bundles already published under `data/synthetic/` stay loadable.
- **Deterministic.** `materialize(plan, snapshot)` is a pure function. The same
  seed and snapshot rebuild byte-identically; the determinism check enforces it.
- **Fail-closed.** Every rename count is asserted at build time, and the leak scan
  requires zero canonical symbols in the output — including in filenames.

## How names are chosen

Renaming happens at the level of **identity tokens**, not whole product names. The
canonical line reuses a small vocabulary (`Green` in nine product names, `Gold` in
eight), so renaming products independently would split families that belong
together. Mapping the shared token once keeps `Gold Account` and `Gold Plus
Account` in one family and makes composed names (`Business Silver Rewards Card`)
follow their base automatically.

Three categories of word are deliberately **left alone**, because they are ordinary
English rather than symbols a model could only have learned here, and renaming them
would damage meaning without removing anything memorisable:

| Category | Examples |
|---|---|
| Structural | `Account`, `Card`, `Rewards`, `Saver`, `Plus`, `Business` |
| Generic modifiers | `Light`, `Dark`, `Sky`, `Navy`, `True`, `World` |
| Plain feature names | `Scheduled Payments`, `Sending Limits`, `QR Transfers` |

Replacement families carry **no implied ordering**. Canonical tiers (Bronze <
Silver < Gold < Platinum < Diamond) let a model rank products from world knowledge
without reading a document; the replacements remove that shortcut, leaving the
documented numbers as the only way to rank them.

Tools keep their noun phrase and change verb plus numeric suffix
(`submit_cash_back_dispute_0589` → `file_cash_back_dispute_9368`). A discoverable
tool's docstring is parsed at runtime into the definition the agent sees, so the
name must keep describing what the tool does — while still forcing the exact
callable name to be retrieved rather than recalled. Infrastructure tools
(`KB_search`, `grep`, `unlock_discoverable_agent_tool`, …) are frozen: the harness
depends on them by literal string.

## Usage

```bash
# 1. Freeze the canonical corpus (read-only; safe while other jobs use it)
tau3 perturb snapshot

# 2. Assign names. Commit the plan and its mapping.csv — they are the audit trail.
tau3 perturb plan --seed 20260909 --output configs/perturb/plans/train-a.plan.json

# A sibling variant that must share no symbol with train-a:
tau3 perturb plan --seed 20260910 --output configs/perturb/plans/eval-a.plan.json \
                  --exclude-plan configs/perturb/plans/train-a.plan.json

# 3. Build
tau3 perturb materialize --plan configs/perturb/plans/train-a.plan.json

# 4. Check: leaks, imports, tool sets, retrieval isolation, determinism, disjointness
tau3 perturb verify --variant data/variants/<id> \
                    --other data/variants/<sibling>

tau3 perturb list
```

## Verification stages

| Stage | What it proves |
|---|---|
| `leak` | No canonical symbol the plan renamed survives anywhere, filenames included |
| `structure` | Modules parse, the package imports cleanly, and the live tool set is exactly the renamed one |
| `retrieval_isolation` | A variant built *after* the canonical domain in one process still indexes its own documents |
| `determinism` | Rebuilding from the same plan and snapshot is byte-identical |
| `disjoint` | Sibling variants share no new symbol |

`retrieval_isolation` guards a real hazard: `tau3.knowledge.embeddings_cache` holds
one process-global, unkeyed document list, so a variant built second would
otherwise index the canonical documents — silently, and looking like unusually good
retrieval. Materialization replaces that cache with a variant-local one in the
generated `retrieval.py`, leaving the canonical module untouched.

## Two matching rules worth knowing

**Boundary-aware, longest-first, single pass.** All spellings compile into one
alternation ordered by descending length, substituted in one pass. So `Rho-Bank+`
beats `Rho-Bank` beats `Rho`; `Rho` never matches inside `Rhode`; and no
replacement can feed another, which makes even a swap well defined.

**JSON is matched decoded, never as raw text.** In raw JSON, `\n` is a backslash
followed by the letter `n` — a word character — which defeats the boundary in front
of the next token and silently undercounts it. Document ids get the same treatment
in reverse: a product slug inside `doc_checking_accounts_light_green_account_001`
is surrounded by underscores, so it is correctly *not* matched there, and each
document id is renamed as a concept of its own.

## Status

Milestone 1 (names) is implemented: brand, products, tools, document ids and the
`rho_bank_subscription` identifier. Numeric jitter, entity ids and distractor
documents are milestones 2 and 3 — see `/root/.claude/plans/task-hack-fancy-key.md`.
Numeric jitter must be rank-preserving: DB-hash replay compares stored gold actions,
so a jitter that flips which product wins would still replay to reward 1 while its
correct answer had silently changed.
