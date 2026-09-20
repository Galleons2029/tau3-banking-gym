# tau3-AA training task synthesis

This pipeline targets the installed `banking_knowledge` environment. It holds
the knowledge base and business tools fixed and generates new customers,
account states, goals and reference actions. All four LLM roles use
`openai/GLM5.3-agentic-qs-h20` by default. The `openai/` prefix selects LiteLLM's
OpenAI-compatible adapter; the endpoint receives model `GLM5.3-agentic-qs-h20`.
Use the existing `.env` endpoint and credentials; no secrets belong in bundles.

## Run the pilot

Install `knowledge`, `gym` and `dev` extras. From the repository root:

```bash
tau3 synthesize build-catalog --config configs/synthesis/tau3-aa.yaml
tau3 synthesize generate --config configs/synthesis/tau3-aa.yaml \
  --num-tasks 20 --output data/synthetic/tau3-aa/pilot
tau3 synthesize validate --bundle data/synthetic/tau3-aa/pilot --resume
tau3 synthesize export --bundle data/synthetic/tau3-aa/pilot
```

Review `report.json`, `catalog.json` (including exclusions), candidate checks
and saved conversations before scaling. For the larger run, use a fresh output
directory and `--num-tasks 200`. Generation is balanced across selection,
cashback, credit-limit and ordering families. A bundle may contain fewer admitted
tasks than requested; failures are reported rather than silently admitted.

For an unattended staged run, the following wrapper requires all 20 pilot tasks
to pass admission before starting the 200-task bundle and training-only SFT:

```bash
python scripts/run_synthesis_batch.py --config configs/synthesis/tau3-aa.yaml \
  --pilot-bundle data/synthetic/tau3-aa/pilot \
  --output data/synthetic/tau3-aa/train-200 \
  --status data/synthetic/tau3-aa/job-status.json --collect-sft
```

It stops and reports shortages or repeated service failures. Re-run the same
command to resume after resolving a failure. Do not run it concurrently with
another writer to the same bundle. Generation replaces candidates that failed
both primary trials, up to ten candidates per slot; outages preserve the candidate.

The three shortlisted checking products in the pilot are Light Green, Blue and
Green (checking). The customer explicitly limits the comparison to that shortlist.
The solver checks mobile-deposit limits, early-payday preferences and monthly-fee
caps, then chooses the unique lowest standard monthly fee. This is not a claim
of global optimality across all products. Cashback examples use whole-point,
nonpromotional purchases because fractional rounding needs additional evidence.
Ordering examples consolidate one to three old entry-tier accounts and optionally
open business checking before closures. The catalog records unresolved KB/tool
conflicts; these are not silently repaired in the benchmark.

## Validation and isolation

Generation performs schema/reference checks, independent business-result checks,
strict real-tool replay, deterministic DB comparison and scoring counterexamples.
Tools returning `Error:` are failures even when they do not raise an exception.
Actual benchmark DB scoring must accept the reference trajectory. The validator
clones an unmodified real environment to avoid rebuilding the same retrieval index;
each replay has separate mutable state shared only between its own agent/user tools.

LLMs generate only the opening narrative. Program-bound user facts remain separate
from the private solution, and a separate review checks factual consistency and
leakage. Generation prompts never include original benchmark tasks, reference
actions, gold documents or computed answers.

Admission requires two **bm25_grep** trials and at least one reviewed success.
`KB_search(query)` and `grep(pattern)` each return at most ten whole documents;
grep is case-insensitive by default. No query string or choice of retrieval tool
is prescribed by the reward. Failed primary trials can be diagnosed using
Only `bm25_grep` is run for synthesis validation. Alternative retrieval modes
are outside this pipeline and do not produce task, acceptance, or SFT evidence.

Published tasks use DB reward. This retains the benchmark's existing audit-log
semantics; it does not add a general-purpose temporal or conversational policy
checker. User compliance and leakage are reviewed separately. Two trials provide
only preliminary difficulty information, not a calibrated success probability.

`--offline` on generation renders deterministic development text. `validate
--offline` runs local checks only. **Offline candidates cannot be published or
used as evidence of successful live conversations.** For example:

```bash
tau3 synthesize generate --num-tasks 20 --offline \
  --output data/synthetic/tau3-aa/offline-check
tau3 synthesize validate --bundle data/synthetic/tau3-aa/offline-check --offline
```

## Bundle format and training

Generation writes atomic `candidates/<slot>.json` checkpoints, a `catalog.json`
and a draft `manifest.json`. `generate --resume` requires the same configuration
and count. Validation saves each completed trial and resumes without repeating
completed calls; infrastructure failures are retryable. Provider outages are
checked before scheduling live work. Concurrent processes must use different
bundle directories.
Completed native conversations and the corresponding visible agent context are
saved before quality review, so a review timeout does not repeat a completed
simulation. Diagnostic failures are recorded without reversing primary acceptance.

Export creates `tasks.json`, `split_tasks.json`, `metadata.jsonl` and
`rejected.jsonl`. The manifest records source checksums, resolved retrieval settings,
models, seed, grouping and artifact hashes. It changes to `published` only after
admission. The loader rejects draft, altered, incompatible or malformed bundles.

The `base` split contains all admitted tasks. Train/validation target 80/20 within
each family, keeping each structural group intact. Whole-group rounding can change
the exact counts; inspect the exported split rather than assuming 160/40. This
tests generalization between scenarios in the same fixed banking world.

```bash
tau3 run --domain banking_knowledge \
  --task-bundle data/synthetic/tau3-aa/pilot --task-split-name train \
  --agent-llm openai/GLM5.3-agentic-qs-h20 \
  --user-llm openai/GLM5.3-agentic-qs-h20
```

The runner defaults to the bundle's recorded retrieval mode. Explicit retrieval
overrides are outside the standard pipeline and are saved in run metadata. Existing
registry task loading remains available when no bundle is specified.

```python
from tau3.synthesis.bundle import load_task_bundle
from tau3.gym.gym_agent import AgentGymEnv

bundle = "data/synthetic/tau3-aa/pilot"
task = load_task_bundle(bundle, "train")[0]
env = AgentGymEnv(
    domain="banking_knowledge", task_id=task.id,
    task_bundle=bundle, task_split_name="train",
    user_llm="openai/GLM5.3-agentic-qs-h20",
)
```

## SFT

```bash
tau3 synthesize collect-sft --bundle data/synthetic/tau3-aa/pilot --split train
```

Only training tasks are sampled, using bm25_grep. Up to four attempts retain at
most two reviewed successes per task. The pipeline saves native `SimulationRun`
files and exports `sft.jsonl` with tools, messages, assistant weights and an
explicit assistant-only `loss_mask`. The export uses the teacher's actual message
state, which excludes the user's private tool calls/results and evaluator data.
Ordinary retrieved text
remains in the teacher's visible context.

Provider-reported costs and token usage are saved when available. Missing pricing
does not mean an API call was free. No external training job is launched.

SFT persists a seed/attempt start marker before constructing a simulation. An
interrupted or failed simulation consumes its slot across resumes; it is not
silently sampled again under the same seed. A saved completed native run may
resume quality review without consuming another simulation. Changed capture
inputs fail instead of resampling. The delivery audit requires matching start and
completion records, so unresolved review work cannot silently pass acceptance.

Run the separate read-only artifact audit before reporting delivery acceptance:

```bash
python scripts/audit_synthesis_bundle.py --bundle data/synthetic/tau3-aa/train-200-v3
# Pilot publication, without requiring its deliberately absent SFT export:
python scripts/audit_synthesis_bundle.py --bundle data/synthetic/tau3-aa/pilot-v3 --expected 20 --rl-only
```

The audit checks publication integrity, balanced families, grouped splits,
resolved retrieval settings, configured model roles, saved strict-check evidence,
two distinct primary seeds, native rewards and review flags. SFT rows must equal
their saved actual agent-visible captures and match retained successful training
trials, with assistant-only masks, at most four attempt slots and two successes.
Missing SFT coverage fails this delivery audit even if the collection process
finished. The audit verifies artifacts; it does not replace business-rule review,
strict replay or the required conversation quality assessment. Coverage exclusions
are reported explicitly and must remain visible in the final delivery report.

## Fixed-environment repair revision

The `fixed-tools-v3` synthesis revision does not modify official domain tools,
agent prompts, retrieval, scoring, or Gym/runner behavior. Under the installed
submit/approve implementation, a credit-limit approval attempts to insert the
same request ID as the pending submission; the insert fails while the card limit
changes. This incompatible branch is explicitly quarantined. The credit-limit
family currently samples denial branches only. Approval coverage is unavailable
under the fixed environment and is not claimed by the bundle.

Structured generation, review and probes use OpenAI `response_format` with
`json_schema` and `strict: true` through the existing LiteLLM adapter. This format
is never injected into agent/user dialogue calls. Generator and judge overrides
are configured separately with `generator_llm_args` and `judge_llm_args`.
Responses, finish reasons, token usage and elapsed time are saved under
`llm_calls/` before parsing; credentials and HTTP headers are not saved there.
`structured_attempts` bounds service/format retries independently of business
candidates. A malformed model response preserves the current skeleton. Exhausted
business slots stop the batch before further expensive dialogue validation.

To resume a pre-repair draft, first stop its writer, then fork it:

```bash
tau3 synthesize repair-bundle --source data/synthetic/tau3-aa/pilot-v2 \
  --output data/synthetic/tau3-aa/pilot-v3 \
  --config configs/synthesis/tau3-aa.yaml
python scripts/run_synthesis_batch.py --config configs/synthesis/tau3-aa.yaml \
  --pilot-bundle data/synthetic/tau3-aa/pilot-v3 \
  --output data/synthetic/tau3-aa/train-200-v3 \
  --status data/synthetic/tau3-aa/job-status.json --collect-sft
```

The source draft is preserved. The fork retains compatible tasks, completed trials
and visible-context captures; conflicting approvals go to `quarantine/` and their
slots are regenerated. It records the parent manifest and a new synthesis hash,
while retaining the same environment hash. New candidates have a revision-specific
ID namespace. Completed compatible trials remain reusable; affected checks run
again during validation. The batch status includes a 30-second heartbeat with
checkpoint counts, separate from the stage start timestamp.

User-behavior review projects agent-private `KB_search`/`grep` result bodies to
explicit omission markers with content hashes. Those bodies were never visible
to the customer, and bank-solution correctness is evaluated separately. The
projection retains every user message, assistant statement, tool call, business
tool result, and private user tool result in order. The complete native simulation,
actual agent context, DB evaluator inputs and SFT context remain unchanged.
This avoids repeatedly sending large retrieved documents to the customer-behavior
judge while preserving the evidence needed for leakage and consistency checks.

The catalog also includes Light Green's documented 13–24 age eligibility.
Selection uses age computed from the customer's birth date at 2025-11-14; low
mobile-deposit scenarios include the age evidence document. Repair quarantines
old selections whose solution or evidence omitted this condition. New customer
instructions require exact numeric bounds. A conservative deterministic review
check rejects recognizable changed daily-mobile-deposit requirements (for example,
2501 restated as "at least $2,500 a day"), even when the model judge approves.
Other paraphrases remain subject to the model review; this check is not a claim
of complete natural-language numeric verification.

Reference actions must use the agent's exposed tool interface. Hidden mutating
methods are compiled through unlock and discovery calls so the official audit
records match actual execution. Repair can regrade preserved conversations after
this reference-only transformation using the unchanged official evaluator with
strict replay. Previous references and scores are archived; user instructions,
initial state and conversation messages are preserved. Ordering scenarios instead
require regeneration: the user must explicitly request the exact recorded closure
reason, since the official DB evaluator compares that field verbatim.

DB failures and deterministic user-fact failures skip model quality review; they
cannot be retained. Successful candidates still require the full review. A
structured response ending with `finish_reason=length` may have exhausted its
budget in reasoning before producing JSON. Only that condition increases the next
structured attempt's output/timeout budgets, bounded by `structured_token_limit`
and `structured_timeout_limit`; effective budgets are saved with raw responses.

If the final allowed dialogue-review attempt still ends at the output limit,
the trajectory is recorded as quality-inconclusive, with `review=null` and
`passed=false`. Its DB reward and native messages remain intact. This differs
from a transport timeout: an inconclusive trajectory can be rejected by the
existing bounded candidate-selection process, while service failures remain
retryable. A missing quality verdict is never replaced with a passing verdict.

Combined closure/business-opening scenarios require the customer to disclose all
requested operations in the first message, before authorizing account changes.
The customer may still prefer closures first; no correct execution order is
included in this disclosure instruction. This prevents a delayed business request
from making an otherwise solvable scenario impossible after irreversible closure.

One in every five ordering slots retains a combined opening/closure requirement
across candidate replacements. This preserves at least one combined scenario in
the 20-task pilot and ten in the 200-task bundle. Other ordering slots can sample
either variant. The manifest records the required minimum; publication and the
independent delivery audit reject a bundle that loses this coverage.

## V2 synthetic seed worlds

`tau3 synthesize world-tasks` is the V2 adapter. The legacy `generate`, `validate`,
`export` and `collect-sft` commands above still target the fixed original banking
world. The V2 adapter generates, verifies and publishes in one resumable command:

```bash
tau3 synthesize world-tasks --world WORLD --readiness READINESS \
  --config configs/worldgen/v2-verification-transport-high.yaml \
  --num-tasks 20 --max-calls 600 --output PILOT
# Resume the exact same declared experiment after interruption:
tau3 synthesize world-tasks --world WORLD --readiness READINESS \
  --config configs/worldgen/v2-verification-transport-high.yaml \
  --num-tasks 20 --max-calls 600 --output PILOT --resume
```

The source must have current independent readiness evidence. Offline/foundation
certificates cannot unlock task synthesis. New task expressions retain the seed's
customer state, business structures and private goal criteria. Numeric literals
and identifiers are checked mechanically; another model checks equivalence and
absence of added solutions. Each candidate is solved independently by both models
using only its actual public request and the public world. Exact selected source
spans and independent calculations are verified before private outcome checks.

Each slot has at most three candidates, retaining its seed business scenario when
replacing a rejected expression. Private failure details are recorded, never sent
back to generation or solving. Every candidate that clears blind checks gets two
real BM25 conversations, one per configured agent model, both of which must finish
normally with reward 1. Rejected native runs are retained, not overwritten with
resampled successes. Service/budget failures stop work; incomplete evidence never
publishes. A quota shortfall leaves the bundle draft. A successful published bundle
can be loaded with `--domain banking_synth --task-bundle PILOT` while
`TAU3_SYNTH_WORLD` points to the matching seed world.

`--max-calls` bounds generation, equivalence review and blind-verification calls.
Conversations have a separate bound of six rollouts per requested slot, each with
the fixed settings' step and time limits. The manifest records both budgets,
source/adapter fingerprints, exact task hashes and validation evidence. Seed,
implementation, source readiness or evidence changes invalidate publication.

Task IDs include a stable namespace for the output batch, so separate batches do
not reuse the pilot's IDs. A larger batch also rejects exact request expressions
already present in its required pilot. This is exact deduplication, not a claim
that linguistic variants introduce new business goals.

This adapter expands linguistic expressions of existing business scenarios; it
does not claim new customer states or new business-goal diversity. It prioritizes
distinct existing structures, and keeps every expression of the same seed task in
the same split. Bundles inherit the seed's structural train/test partition, so a
structure cannot switch partitions between pilot and later batches. After a
complete 20-task pilot, a separate 200-task experiment can
use `--num-tasks 200 --max-calls 6000 --pilot-bundle PILOT` with a fresh output directory. Large-task
readiness must be supported by an actual passing pilot; these commands alone are
not an acceptance certificate. V2 SFT collection is not implemented here.


The staged wrapper runs the required pilot, then prepares a guarded launch command
without starting the larger batch when `--prepare-only` is selected:

```bash
python scripts/run_world_task_synthesis.py --world WORLD --readiness READINESS \
  --config configs/worldgen/v2-verification-transport-high.yaml \
  --pilot PILOT --output BATCH_200 --num-tasks 200 --prepare-only
```

Optional `--local-world MIRROR` requires byte-identical seed artifacts and speeds
validation on a local filesystem. Canonical seed paths remain in the manifest.
Removing `--prepare-only` runs the bounded batch after the pilot passes. The CLI
also enforces the published 20-task pilot requirement for batches above 20.

The complete qualification driver copies existing snapshots, checks the small
gates, runs the full seed matrix, admits the pilot, and prepares the launch file:

```bash
python scripts/qualify_world_task_pipeline.py \
  --source data/synthetic/worldgen-v2-source-high \
  --output data/synthetic/worldgen-v2-transport-high \
  --config configs/worldgen/v2-verification-transport-high.yaml
```

`stage-state.json` records progress. `READY` requires all stages and the real
20-task pilot to pass. Interrupted or failed evidence remains in its original
directory. The transport configuration allows at most three physical audit
attempts for timeouts, connection failures, rate limits or HTTP 5xx. Every attempt
is reserved, recorded and charged to the same budget. Returned judgments,
truncation and content errors do not trigger transport retries. Calibration
reports show recovered transport failures separately from scoring validity.

Full seed qualification partitions blind tasks across at most four isolated
processes, with disjoint task lists and a combined 1000-call budget. The driver
retains shard records, refuses incomplete shards or overlapping requests, and
uses the original blind checker to replay the merged raw responses. The two
online model/retrieval combinations also run in separate processes; each retains
the same task membership, model settings and two-conversation concurrency limit.
`--no-parallel-online` on the independent-validation driver selects sequential
retrieval conditions when the gateway needs lower concurrency.

After a checker-only change, an explicitly selected `--reuse-blind-from OLD_RUN`
can import normally completed raw blind responses into a fresh experiment. Public
artifacts, settings, audit kind and request hashes must match. Invalid returned
content is retained too; source identities are recorded for every reused response.
The new checker reconstructs each request and recomputes all proof results. This
does not import PASS reports, calibration judgments or online rollout acceptance.
Public solving permits four candidate attempts and eight solver/reviewer calls,
still subject to the experiment-wide physical call budget. Missing citation IDs
and invalid calculations produce public protocol feedback, never private goals.

### Shared LLM concurrency for V2 qualification and task synthesis

Set both variables in the launcher environment to opt into a shared request pool:

```bash
export TAU3_LLM_CONCURRENCY=128
export TAU3_LLM_POOL=/tmp/tau3-worldgen-llm-pool-128
```

All qualification children inherit the same pool. The four full-matrix online
workers use 32 conversation slots each. Candidate generation and public blind
verification run across tasks with up to 128 workers; the subsequent task rollout
round uses up to 128 conversation slots. Agent, user and judge completions acquire
the same host-wide pool, so 128 is the total request ceiling, not a per-model limit.
Small batches and local tool work can leave slots idle. SDK retries retain their
slot. `status.json` in the pool directory records active leases, admitted/released
calls and the observed peak; these count completion scopes, including SDK retries,
not individual retry HTTP attempts. Dead processes' leases are reclaimed on the
next pool access. Use a local filesystem and the same directory on the same host
for every participating process; a pool's configured limit cannot change in place.

Concurrent task audits lock request identities and reserve the shared physical-call
budget before network I/O. Identical requests reuse one saved response; incomplete
requests remain inconclusive. Candidate deduplication follows slot order, independent
of completion order. A bundle lock prevents overlapping writers during resume.
The generated `launch.json` command preserves the request-pool environment.
`--prepare-only` still stops after admitting the pilot and preparing the batch command.

To resume full qualification after a controlled worker shutdown, use the original
output/config/source with `scripts/qualify_world_task_pipeline.py --resume-full`.
This skips snapshot refresh and response import, but the full driver rechecks the
frozen smoke gates and native checkpoints before continuing. Keep the shutdown
record and snapshots; unfinished conversations must not be counted as passed.

### Streaming V2 task admission and SFT

`run_world_streaming.py` adds an independent streaming producer without changing
existing seed or pilot certificates. Its `--mode all` supervises two processes:
32 pilot teacher slots and 96 task workers, all sharing the configured 128-request
pool. The pilot process samples the two teachers for the 16 published training
pilot tasks. It exports up to 32 successful examples to `pilot-sft/sft.jsonl` and
atomic shards. Validation-split tasks never enter this export.

The task process retargets the frozen 8k draft to the first 6,000 source slots,
retaining each slot's original task namespace, business anchor, public prompts and
seed split. It imports exact raw model responses lazily, records provenance and
charges imported attempts against the 180,000-call audit budget. Negative or
invalid returned responses are retained too. Only old requests without returned
content may be retried under the explicit retarget transition; their original
records remain in the source and in the new interruption ledger.

Each task independently advances through expression checks, two public-only
solutions/cross-reviews, private outcome checks and two actual BM25 conversations.
Both conversations must pass before `admitted/NNNNNN.json` is published. For train
tasks, those same conversations supply teacher samples, avoiding a second set of
identical-purpose rollouts. Capture saves the teacher's actual system/history/tools
before evaluation; it does not reconstruct teacher context from the omniscient
simulation transcript. Each SFT row includes assistant-only `loss_mask`, model,
seed and evidence references. A started but unfinished sampling or grading attempt
is not silently resampled.

Training shards under `sft/shards/*.jsonl` are written atomically as tasks are
admitted and can be consumed before the entire job finishes. The new stream has
its own producer identity and per-task proof format; its aggregate `tasks.json`
is not advertised as a legacy `--task-bundle` publication certificate. Full target
completion and partial delivery are distinguished in `progress.json` and
`manifest.json`. Infrastructure/budget inconclusives stop new queued tasks, while
already running tasks retain their evidence. Missing quota never becomes PASS.

```bash
TAU3_LLM_CONCURRENCY=128 \
TAU3_LLM_POOL=/tmp/tau3-worldgen-llm-128-public-repair \
.venv/bin/python scripts/run_world_streaming.py \
  --root data/synthetic/worldgen-v2-public-repair-high \
  --config configs/worldgen/v2-verification-transport-high.yaml \
  --output data/synthetic/worldgen-v2-public-repair-high/task-batch-6000 \
  --count 6000
```
