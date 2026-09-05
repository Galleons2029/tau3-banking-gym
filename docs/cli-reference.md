# CLI Reference

The `tau2` command provides a unified interface for all τ-bench functionality. Use `tau2 <command> --help` to see full details for any command.

Run `tau2 intro` (or just `tau2`) to see an overview of available domains, commands, and a quick-start guide directly in the terminal.

## `tau2 run` — Run Evaluations

Run agent evaluations across different communication modes.

### Basic Usage

```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --num-tasks <task_count>
```

### Common Options

| Option | Description |
|--------|-------------|
| `--domain`, `-d` | Domain to evaluate: `banking_knowledge`, `mock` |
| `--agent-llm` | LLM model for the agent |
| `--user-llm` | LLM model for the user simulator |
| `--agent-llm-args` | JSON dict of extra args for agent LLM (e.g. `'{"temperature": 0.5}'`) |
| `--user-llm-args` | JSON dict of extra args for user LLM |
| `--agent` | Agent implementation to use (default: `llm_agent`) |
| `--user` | User simulator implementation to use (default: `user_simulator`) |
| `--num-trials` | Number of evaluation trials (default: `1`) |
| `--num-tasks` | Number of tasks to evaluate (omit for all tasks) |
| `--task-ids` | Specific task IDs to evaluate |
| `--task-split-name` | Task split to use (default: `base`) |
| `--task-set-name` | Task set to use (default: domain default) |
| `--max-steps` | Maximum simulation steps (default: `200`) |
| `--max-errors` | Maximum consecutive tool errors allowed (default: `10`) |
| `--max-concurrency` | Maximum concurrent simulations (default: `3`) |
| `--seed` | Random seed for reproducibility (default: `300`) |
| `--save-to` | Custom output directory name (saved under `data/simulations/`) |
| `--log-level` | Log level (default: `ERROR`) |
| `--verbose-logs` | Save detailed logs (LLM calls, per-task logs) |
| `--llm-log-mode` | LLM log mode when `--verbose-logs` is on: `all` or `latest` (default: `latest`) |
| `--max-retries` | Max retries for failed tasks (default: `3`) |
| `--retry-delay` | Delay in seconds between retries (default: `1.0`) |
| `--enforce-communication-protocol` | Enforce protocol rules (e.g. no mixed text + tool call messages) |
| `--user-persona` | User persona config as JSON dict |
| `--auto-resume` | Automatically resume from existing save file without prompting |
| `--auto-review` | Automatically run LLM conversation review after each simulation |
| `--review-mode` | Review mode when `--auto-review` is on: `full` or `user` (default: `full`) |
| `--review-model` | LLM model to use for review calls (default: `claude-opus-4-5`) |
| `--timeout` | Maximum wallclock time in seconds per simulation (no timeout by default) |
| `--retrieval-config` | Retrieval configuration for `banking_knowledge` domain (e.g., `alltools`, `bm25`, `terminal_use`) |
| `--retrieval-config-kwargs` | JSON arguments for the retrieval config constructor (e.g., `'{"top_k": 10}'`) |

### Examples

```bash
# Standard text evaluation
tau2 run --domain banking_knowledge --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 1 --num-tasks 5

# Knowledge retrieval with BM25
tau2 run --domain banking_knowledge --retrieval-config bm25 \
  --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 5

# Knowledge retrieval with embeddings and reranker
tau2 run --domain banking_knowledge --retrieval-config openai_embeddings_reranker \
  --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 5
```


---

## `tau2 play` — Interactive Play Mode

Experience τ-bench interactively from either perspective.

```bash
tau2 play
```

Play mode allows you to:
- **Play as Agent**: Manually control the agent's responses and tool calls
- **Play as User**: Control the user while an LLM agent handles requests (available in domains with user tools, such as `banking_knowledge`)
- **Understand tasks** by walking through scenarios step-by-step
- **Test strategies** before implementing them in code
- **Choose task splits** to practice on training data or test on held-out tasks

See the [Gym Documentation](../src/tau2/gym/README.md) for using the gymnasium interface programmatically.

---

## `tau2 view` — View Results

Browse and analyze simulation results.

```bash
tau2 view
```

| Option | Description |
|--------|-------------|
| `--dir` | Directory containing simulation files (defaults to `data/simulations/`) |
| `--file` | Path to a specific results file to view |
| `--only-show-failed` | Only show failed tasks |
| `--only-show-all-failed` | Only show tasks that failed in all trials |

---

## `tau2 domain` — View Domain Documentation

View domain policy and API documentation.

```bash
tau2 domain <domain>
```

Then visit http://127.0.0.1:8004/redoc to see the domain policy and available tools.

---

## `tau2 check-data` — Check Data Configuration

Verify that your data directory is properly configured.

```bash
tau2 check-data
```

---

## `tau2 start` — Start All Servers

Start all domain servers.

```bash
tau2 start
```

---

## `tau2 evaluate-trajs` — Evaluate Trajectories

Re-evaluate trajectory files and optionally update rewards.

```bash
tau2 evaluate-trajs <paths...>
```

| Option | Description |
|--------|-------------|
| `<paths>` | Paths to trajectory files, directories, or glob patterns |
| `-o`, `--output-dir` | Directory to save updated trajectories. If omitted, only displays metrics |

---

## `tau2 review` — LLM Conversation Review

Run LLM-based review on simulation results to detect agent and/or user errors.

```bash
tau2 review <path>
```

| Option | Description |
|--------|-------------|
| `<path>` | Path to a `results.json` file or directory containing them |
| `-m`, `--mode` | Review mode: `full` (agent + user, default) or `user` (user simulator only) |
| `-o`, `--output` | Output path for reviewed results (single file only) |
| `--interruption-enabled` | Flag indicating interruption was enabled in these simulations |
| `--show-details` | Show detailed review for each simulation |
| `-c`, `--max-concurrency` | Max concurrent reviews (default: `32`) |
| `--limit` | Limit review to first N simulations |
| `--task-ids` | Only review simulations for these task IDs |
| `--log-llm` | Log LLM request/response for each review call |
| `--review-model` | LLM model to use for review calls (default: `claude-opus-4-5`) |

---

## `tau2 convert-results` — Convert Results Format

Convert simulation results between monolithic JSON and directory-based formats.

```bash
tau2 convert-results <path> [--to {json,dir}] [--no-backup]
```

| Option | Description |
|--------|-------------|
| `<path>` | Path to a `results.json` file or directory containing one |
| `--to` | Target format: `json` (monolithic) or `dir` (directory with individual sim files). If omitted, converts to the opposite of the current format |
| `--no-backup` | Skip creating a backup before conversion |

Runs default to monolithic JSON. Use this command to convert to or from the directory-based format when needed.

---

## `tau2 leaderboard` — View Leaderboard

Show the τ-bench leaderboard in the terminal.

```bash
tau2 leaderboard
```

| Option | Description |
|--------|-------------|
| `--domain`, `-d` | Show leaderboard for a specific domain: `retail`, `airline`, `telecom`, or `banking_knowledge` |
| `--metric`, `-m` | Metric to rank by: `pass_1`, `pass_2`, `pass_3`, `pass_4`, `cost` (default: `pass_1`) |
| `--limit`, `-n` | Limit the number of entries shown |

---

## `tau2 submit` — Leaderboard Submission

See the full [Leaderboard Submission Guide](leaderboard-submission.md).

```bash
# Prepare a submission
tau2 submit prepare <paths...> --output ./my_submission


# Skip trajectory verification during preparation
tau2 submit prepare <paths...> --output ./my_submission --no-verify

# Validate a submission
tau2 submit validate <submission_dir> [--mode public|private]

# Verify trajectory files
tau2 submit verify-trajs <paths...> [--mode public|private]
```

---

## Environment CLI (beta)

An interactive CLI for directly querying and testing domain environments.

```bash
make env-cli
```

**Commands:**
- `:q` — quit
- `:d` — change domain
- `:n` — start new session (clears history)

**Example:**
```bash
$ make env-cli

Welcome to the Environment CLI!
Connected to banking_knowledge domain.

Query (:n new session, :d change domain, :q quit)> What is the annual fee on the premium credit card?
Assistant: Let me look that up in the knowledge base...
```

Useful for testing domain tools, debugging environment responses, and exploring domain functionality without starting the full server stack.

---

## Running Tests

```bash
make test              # Core tests (requires: uv sync --extra dev)
make test-knowledge    # Banking knowledge tests (requires: uv sync --extra dev --extra knowledge)
make test-gym          # Gymnasium tests (requires: uv sync --extra dev --extra gym)
make test-all          # All tests (requires: uv sync --all-extras)
```

---

## Advanced: Alternative agent implementations

`--agent` selects the agent implementation. Besides the default `llm_agent`:

- `llm_agent_gt` — the agent is handed an oracle plan, removing the need for
  action planning. Only applies to tasks that carry a reference trajectory.
- `llm_agent_solo` — the agent is given all tools and information upfront and
  runs without a user (`--user dummy_user`). Note that `banking_knowledge`
  rejects solo mode, so this currently only applies to `mock`.

```bash
tau2 run \
  --domain banking_knowledge \
  --agent llm_agent_gt \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1
```

> The ablation studies from the original τ-bench paper (no-user, oracle-plan
> and workflow-policy runs on the `telecom` domain) are not reproducible here:
> this fork bundles only `banking_knowledge` and `mock`.
