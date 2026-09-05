# Running Simulations

This guide covers the `tau3.runner` API for running simulations at different levels of control. Whether you want a one-liner CLI command or full programmatic control over every component, the runner framework has you covered.

## Architecture Overview

The runner framework is organized into three layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| **Layer 1** | `runner.simulation` | Execute a pre-built orchestrator. No registry, no config. |
| **Layer 2** | `runner.build` | Build instances (environment, agent, user, orchestrator) from names/config via the registry. |
| **Layer 3** | `runner.batch` | Batch execution with concurrency, checkpointing, retries, logging, and side effects. |

Each layer builds on the one below it, but you can enter at any level depending on your needs.

---

## Level 1: CLI (Simplest)

The fastest way to run simulations is through the `tau3 run` command:

```bash
# Run all banking_knowledge tasks with GPT-4.1
tau3 run --domain banking_knowledge --agent llm_agent --agent-llm openai/gpt-4.1

# Run specific tasks with multiple trials
tau3 run --domain banking_knowledge --agent llm_agent --agent-llm openai/gpt-4.1 \
    --task-ids task_001 task_002 --num-trials 3

# Run with concurrency and auto-resume
tau3 run --domain banking_knowledge --agent llm_agent --agent-llm openai/gpt-4.1 \
    --max-concurrency 4 --auto-resume
```

See `tau3 run --help` or [CLI Reference](cli-reference.md) for all options.

---

## Level 2: Config-Based Python API (Intermediate)

For programmatic use, create a `TextRunConfig` and call `run_domain()`:

### Text (half-duplex) simulations

```python
from tau3 import TextRunConfig
from tau3.runner import run_domain

config = TextRunConfig(
    domain="banking_knowledge",
    agent="llm_agent",
    llm_agent="openai/gpt-4.1",
    llm_user="openai/gpt-4.1-mini",
    num_trials=3,
    max_concurrency=4,
    seed=42,
)

results = run_domain(config)
# compute_metrics() returns an AgentMetrics object with avg_reward, pass_hat_ks, etc.
from tau3.metrics.agent_metrics import compute_metrics
metrics = compute_metrics(results)
print(f"Average reward: {metrics.avg_reward}")
```

### Knowledge retrieval simulations

```python
from tau3 import TextRunConfig
from tau3.runner import run_domain

config = TextRunConfig(
    domain="banking_knowledge",
    agent="llm_agent",
    llm_agent="openai/gpt-4.1",
    llm_user="openai/gpt-4.1",
    retrieval_config="alltools",  # or "bm25", "openai_embeddings", "terminal_use", etc.
    num_trials=3,
)

results = run_domain(config)
```

`TextRunConfig` extends `BaseRunConfig`, which carries fields like `domain`, `num_trials`, and `seed`. The `RunConfig` type alias resolves to `TextRunConfig`.

This handles everything: task loading, filtering, concurrency, checkpointing, metrics display.

### Running specific tasks

```python
from pathlib import Path
from tau3 import TextRunConfig
from tau3.runner import get_tasks, run_tasks

config = TextRunConfig(
    domain="banking_knowledge",
    agent="llm_agent",
    llm_agent="openai/gpt-4.1",
)

# Load and filter tasks manually
tasks = get_tasks("banking_knowledge", task_ids=["task_001", "task_002"])

# Run with custom save path
results = run_tasks(
    config,
    tasks,
    save_path=Path("my_results/results.json"),
    save_dir=Path("my_results/"),
)
```

### Running a single task

```python
from tau3 import TextRunConfig
from tau3.runner import run_single_task, get_tasks

config = TextRunConfig(domain="banking_knowledge", agent="llm_agent", llm_agent="openai/gpt-4.1")
tasks = get_tasks("banking_knowledge")

result = run_single_task(config, tasks[0], seed=42)
print(f"Task {result.task_id}: reward={result.reward_info.reward}")
```

---

## Level 3: Instance-Based Python API (Power User)

For maximum control, build instances yourself and use `run_simulation()`. This is ideal for:
- Custom agents or environments not in the registry
- Fine-grained control over agent/user construction
- Testing and development
- Integration into custom pipelines

### Using build helpers with the registry

```python
from tau3.runner import (
    build_environment,
    build_agent,
    build_user,
    build_orchestrator,
    run_simulation,
    get_tasks,
)
from tau3.evaluator.evaluator import EvaluationType

# Load a task
tasks = get_tasks("banking_knowledge")
task = tasks[0]

# Option A: Build orchestrator from config (uses registry)
from tau3 import TextRunConfig
config = TextRunConfig(domain="banking_knowledge", agent="llm_agent", llm_agent="openai/gpt-4.1")
orchestrator = build_orchestrator(config, task, seed=42)
result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)

# Option B: Build components individually
env = build_environment("banking_knowledge")
agent = build_agent("llm_agent", env, llm="openai/gpt-4.1")
user = build_user("user_simulator", env, task, llm="openai/gpt-4.1-mini")
```

### Fully custom instances (no registry needed)

```python
from tau3.runner import run_simulation
from tau3.orchestrator.orchestrator import Orchestrator
from tau3.user.user_simulator import UserSimulator

# Your custom environment and agent
env = MyCustomEnvironment()
agent = MyCustomAgent(tools=env.get_tools(), domain_policy=env.get_policy())

# Standard user simulator
user = UserSimulator(
    tools=env.get_user_tools(),
    instructions=str(task.user_scenario),
    llm="openai/gpt-4.1-mini",
)

# Wire everything into an orchestrator
orchestrator = Orchestrator(
    domain="my_domain",
    agent=agent,
    user=user,
    environment=env,
    task=task,
    max_steps=100,
    max_errors=10,
    seed=42,
)

# Run and evaluate
result = run_simulation(orchestrator)
print(f"Reward: {result.reward_info.reward}")
```

---

## API Reference

### Layer 1: Execution

- **`run_simulation(orchestrator, *, evaluation_type=EvaluationType.ALL, env_kwargs=None)`** -- Run a pre-built orchestrator and evaluate the result. Returns `SimulationRun` with `reward_info` attached.

### Layer 2: Build

- **`build_environment(domain, *, solo_mode=False, env_kwargs=None)`** -- Build an environment from a domain name.
- **`build_agent(agent_name, environment, *, llm, llm_args, task, solo_mode)`** -- Build an agent from a registered name.
- **`build_user(user_name, environment, task, *, llm, llm_args, persona_config, solo_mode)`** -- Build a half-duplex user from a registered name.
- **`build_text_orchestrator(config, task, *, seed, simulation_id, user_persona_config)`** -- Build a half-duplex `Orchestrator` from a `TextRunConfig`.
- **`build_orchestrator(config, task, *, seed, simulation_id, user_persona_config)`** -- Thin wrapper over `build_text_orchestrator`.

### Layer 3: Batch Execution

- **`run_domain(config)`** -- Run all tasks for a domain from a `RunConfig`. Handles task loading, filtering, save paths, metrics.
- **`run_tasks(config, tasks, *, save_path, save_dir, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS, console_display)`** -- Run a list of tasks with concurrency, checkpointing, and retries. Note: defaults to `ALL_WITH_NL_ASSERTIONS` (unlike `run_simulation` which defaults to `ALL`).
- **`run_single_task(config, task, *, seed, evaluation_type, save_dir, ...)`** -- Run one task with logging and optional side effects (auto-review).

### Helpers

- **`get_options()`** -- List available domains, agents, users, and task sets.
- **`get_tasks(task_set_name, task_split_name, task_ids, num_tasks)`** -- Load tasks with optional filtering.
- **`load_tasks(task_set_name, task_split_name)`** -- Load raw tasks from the registry.
- **`get_info(config, **overrides)`** -- Create run metadata (`Info` object).
- **`make_run_name(config)`** -- Generate a timestamped run name.

---

## Choosing the Right Level

| Scenario | Recommended Level |
|----------|------------------|
| Quick evaluation from the command line | Level 1 (CLI) |
| Running evals in a Python script | Level 2 (Config-based) |
| Custom agent development and testing | Level 3 (Instance-based) |
| Integrating into a larger pipeline | Level 3 (Instance-based) |
| Hyperparameter sweeps | Level 2 (Config-based) |
| Contributing a new domain/agent | Level 3 for testing, Level 2 for benchmarking |
