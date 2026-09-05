# Getting Started

This guide walks you through installing τ-bench, configuring API keys, and running your first evaluation.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and project manager
- Python 3.12+ (uv will download it automatically if not present)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

### 2. Install τ-bench

```bash
uv sync                        # core only (text-mode: mock)
```

This creates a virtual environment, installs core dependencies from the lockfile, and enables the `tau2` command. The Python version is pinned via `.python-version` (3.12) — uv will download it automatically if needed.

#### Optional extras

Install the extras you need:

```bash
uv sync --extra knowledge      # + banking_knowledge domain (retrieval pipeline)
uv sync --extra gym            # + gymnasium RL interface
uv sync --extra dev            # + pytest, ruff, pre-commit (required for contributing)
uv sync --extra experiments    # + plotting libs for src/experiments/
uv sync --all-extras           # everything
```

> **Note:** If you install without `-e` mode (e.g., `uv pip install .`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:
> ```bash
> export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
> ```

### 3. Verify your installation

```bash
uv run tau2 check-data
```

This checks that your data directory is correctly configured and all required files are present.

## Setting Up API Keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

Copy `.env.example` as `.env` and edit it to include your API keys:

```bash
cp .env.example .env
```

## Running Your First Evaluation

### Standard text-based evaluation (half-duplex)

```bash
tau2 run --domain banking_knowledge --agent-llm gpt-4.1 --user-llm gpt-4.1 \
  --num-trials 1 --num-tasks 5
```

Results are saved in `data/simulations/`.

### Knowledge retrieval evaluation

```bash
tau2 run --domain banking_knowledge --retrieval-config bm25 \
  --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 5
```

See the [Knowledge Retrieval Documentation](../src/tau2/knowledge/README.md) for retrieval configuration options.


> **tip**: for full agent evaluation that matches the original τ-bench methodology, remove `--num-tasks` to evaluate on the complete task set (the `base` split is used by default).

## Simulation Output Structure

Results are stored in one of two formats:

### Monolithic JSON (default)

Simulations produce a single file containing all data:

```
data/simulations/<run_name>/
└── results.json             # Metadata, tasks, and all simulation data
```

### Directory-based format

For large runs, a directory-based format splits simulation data into individual
files for efficient checkpointing and streaming:

```
data/simulations/<run_name>/
├── results.json                        # Metadata and task definitions only
├── simulations/                        # Individual simulation data files
│   ├── sim_0.json
│   ├── sim_1.json
│   └── ...
└── artifacts/                          # Runtime artifacts (with --verbose-logs)
    └── task_<task_id>/
        └── sim_<uuid>/
            ├── sim_status.json         # Simulation status
            ├── task.log                # Per-task log
            └── llm_debug/
                └── *.json              # LLM call logs
```

### Format conversion

You can convert between formats using `tau2 convert-results`:

```bash
# Convert a monolithic JSON to directory format
tau2 convert-results data/simulations/my_run --to dir

# Convert a directory format back to monolithic JSON
tau2 convert-results data/simulations/my_run --to json
```

Both formats are fully supported by `Results.load()`, which auto-detects the format on disk.

## Viewing Results

```bash
tau2 view
```

This allows you to browse simulation files, view agent performance metrics, inspect individual simulations, and view task details.

## Configuration

The framework is configured via [`src/tau2/config.py`](../src/tau2/config.py).

### LLM Call Caching

LLM call caching is disabled by default. To enable it:

1. Install the `redis` Python package: `uv pip install redis`
2. Make sure a Redis server is running
3. Update the redis config in `config.py` if necessary
4. Set `LLM_CACHE_ENABLED` to `True` in `config.py`

## Cleanup

To remove all generated files and the virtual environment:

```bash
make clean
```

## Next Steps

- [CLI Reference](cli-reference.md) — all `tau2` commands and options
- [Agent Developer Guide](../src/tau2/agent/README.md) — build and evaluate your own agent
- [Domain Documentation](../src/tau2/domains/README.md) — understand the available domains
- [Orchestrator](../src/tau2/orchestrator/README.md) — half-duplex (turn-based) orchestration
- [Task Schema & Evaluation](evaluation.md) — how a task is scored, what `actions`/`communicate_info`/`reward_basis` actually do
- [Knowledge Retrieval](../src/tau2/knowledge/README.md) — retrieval pipeline setup and configuration for banking_knowledge domain
- [Gym/RL Interface](../src/tau2/gym/README.md) — Gymnasium-compatible environment for RL training
