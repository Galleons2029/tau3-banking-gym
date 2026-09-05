# AGENTS.md — src/tau3/evaluator/

## Overview

The evaluator module scores simulation runs. Each evaluator is a classmethod-only class that takes a task and trajectory, and returns a `RewardInfo`.

## Architecture

### Evaluator Types

| Evaluator | What it checks | `RewardType` |
|-----------|---------------|-------------|
| `EnvironmentEvaluator` | DB state matches gold after replaying actions | `DB`, `ENV_ASSERTION` |
| `ActionEvaluator` | Agent called the right tools with right args | `ACTION` |
| `CommunicateEvaluator` | Agent said required information to user | `COMMUNICATE` |
| `NLAssertionsEvaluator` | LLM-judged natural language assertions (WIP) | `NL_ASSERTION` |

### Evaluator Shape

Every evaluator is a `EvaluatorBase[Message]` subclass operating on `list[Message]`
(the half-duplex trajectory).

### Composition via `evaluate_simulation()`

The `evaluate_simulation()` function in `evaluator.py` is the main entry point. It:
1. Runs the appropriate evaluators based on `EvaluationType`
2. Combines rewards multiplicatively based on the task's `reward_basis`

### `EvaluationType` Controls What Gets Evaluated

- `ALL` — runs ENV + ACTION + COMMUNICATE, but only includes each in the final reward if it's in the task's `reward_basis`
- `ALL_IGNORE_BASIS` — runs all three and always multiplies them together
- `ENV`, `ACTION`, `COMMUNICATE` — run a single evaluator
- `NL_ASSERTIONS`, `ALL_WITH_NL_ASSERTIONS` — include NL assertions (WIP)
- `ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS` — like `ALL_IGNORE_BASIS` but also includes NL assertions (WIP)

## Rules When Modifying This Module

1. **`calculate_reward` is a classmethod**: All evaluators use `@classmethod`. No instance state.

2. **Reward is multiplicative**: The combined reward is the product of component rewards. A 0 in any component zeroes the total.

3. **Handle missing criteria gracefully**: If `task.evaluation_criteria` is `None` or a specific criterion is absent, return `reward=1.0` (not 0). Missing criteria means "not evaluated", not "failed".

4. **Premature termination = 0 reward**: If `simulation.termination_reason` is not `AGENT_STOP` or `USER_STOP`, the simulation gets reward 0 before any evaluators run.

5. **`evaluation_criteria.actions` is *one* reference trajectory, not a per-action requirement**: `EnvironmentEvaluator` replays `actions` on a fresh env to derive the target DB hash, then compares against the predicted DB hash. Other agent trajectories producing an equivalent end state also pass. The agent is only required to match those specific calls when `RewardType.ACTION` is in `reward_basis` (rare; not used by airline/retail/telecom). See `docs/evaluation.md`.

6. **Environment evaluator needs `environment_constructor` and `env_kwargs`**: Unlike other evaluators, `EnvironmentEvaluator.calculate_reward()` takes an `environment_constructor` callable to create fresh environments for gold vs predicted comparison. It also accepts `env_kwargs: dict` for domain-specific parameters (e.g., `retrieval_variant` for `banking_knowledge`). These kwargs are threaded from `evaluate_simulation()` through to both the predicted and gold environment constructors. Always pass `env_kwargs` when calling `calculate_reward()` to ensure domains with custom constructor parameters work correctly.

7. **NL assertions require LLM calls**: `NLAssertionsEvaluator` calls an LLM (configured via `DEFAULT_LLM_NL_ASSERTIONS` in `config.py`). This is experimental/WIP.


Consider these rules if they affect your changes.
