# AGENTS.md — src/tau3/agent/

> See `README.md` for the full developer guide with code examples.
> See `base/README.md` for the protocol base class and LLM config mixin.

## Rules for Working in This Directory

### Base Class

| Communication mode | Base class | Key method |
|-------------------|-----------|------------|
| Half-duplex (turn-based text) | `HalfDuplexAgent` | `generate_next_message()` |

### Constructor Contract

The base class MUST accept: `__init__(self, tools: list[Tool], domain_policy: str)`.
For LLM-powered agents, add `LLMConfigMixin` which provides `llm: str` and `llm_args: dict`.

### MRO (Mixin Inheritance Order) Is Critical

Python MRO determines which `__init__` runs. Follow this exact order:

```python
class MyAgent(
    LLMConfigMixin,             # 1. Config mixins first
    HalfDuplexAgent[StateType]  # 2. Protocol base class last
):
```

Getting this wrong causes subtle bugs where `__init__` arguments silently disappear.

### Message Constraints

- A message MUST have either `content` (text) OR `tool_calls`, NEVER both simultaneously.
- `AssistantMessage.is_tool_call()` returns `True` when `tool_calls` is non-empty.

### State Classes

- Use Pydantic `BaseModel` for all state classes.

### Registration

Every agent needs a factory function and registry entry:

```python
# Factory signature:
def create_agent(tools, domain_policy, **kwargs):
    # kwargs may include: llm, llm_args, task, etc.
    return MyAgent(tools=tools, domain_policy=domain_policy, ...)

# In src/tau3/registry.py:
registry.register_agent_factory(create_my_agent, "my_agent")
```

The name passed to `register_agent_factory` is what users pass via `--agent` on the CLI.

### Stop Signals

- `is_stop()` classmethod checks for stop conditions.

### Testing

- Unit tests: `tests/test_agent.py`
