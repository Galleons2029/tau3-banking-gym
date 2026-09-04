"""
Layer 2: Build functions.

Turn config/names into live instances (environment, agent, user, orchestrator).
Uses the registry for name resolution. Callers who want full control can skip
this layer and construct instances directly.
"""

import uuid
from typing import Optional

from loguru import logger

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import RunConfig, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator
from tau2.user.user_simulator_base import HalfDuplexUser

# =============================================================================
# Low-level build functions (no RunConfig needed)
# =============================================================================


def build_environment(
    domain: str,
    *,
    solo_mode: bool = False,
    env_kwargs: Optional[dict] = None,
) -> Environment:
    """Build an environment from a domain name.

    Uses the registry to resolve the domain name to an environment constructor.

    Args:
        domain: Domain name (e.g., "airline", "retail", "mock").
        solo_mode: If True, environment is built in solo mode (agent gets
            access to both agent and user tools).
        env_kwargs: Additional keyword arguments passed to the environment
            constructor (e.g., retrieval_variant, task for banking_knowledge).

    Returns:
        A fully constructed Environment instance.
    """
    env_constructor = registry.get_env_constructor(domain)
    kwargs = dict(env_kwargs or {})
    if solo_mode:
        kwargs["solo_mode"] = True
    return env_constructor(**kwargs)


def build_agent(
    agent_name: str,
    environment: Environment,
    *,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    task: Optional[Task] = None,
    solo_mode: bool = False,
) -> HalfDuplexAgent:
    """Build an agent from a registered name and an environment.

    Uses the registry to resolve the agent name to a factory function,
    then calls it with the appropriate parameters.

    Args:
        agent_name: Registered agent name (e.g., "llm_agent", "llm_agent_gt",
            or "experimental:my_agent").
        environment: The environment to extract tools and policy from.
        llm: LLM model name for the agent.
        llm_args: LLM arguments for the agent.
        task: The task (required for some agents like llm_agent_gt, llm_agent_solo).
        solo_mode: If True, agent tools include both agent and user tools.

    Returns:
        A fully constructed agent instance.

    Raises:
        ValueError: If the agent name has no factory registered.
    """
    agent_factory = registry.get_agent_factory(agent_name)
    if agent_factory is None:
        raise ValueError(
            f"Agent '{agent_name}' has no factory registered. "
            f"Register a factory with registry.register_agent_factory()."
        )

    # Collect tools from environment
    tools = environment.get_tools()
    if solo_mode:
        try:
            user_tools = environment.get_user_tools()
            if user_tools:
                tools = tools + user_tools
        except Exception:
            pass

    return agent_factory(
        tools=tools,
        domain_policy=environment.get_policy(),
        llm=llm,
        llm_args=llm_args,
        task=task,
    )


def build_user(
    user_name: str,
    environment: Environment,
    task: Task,
    *,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    persona_config: Optional[PersonaConfig] = None,
    solo_mode: bool = False,
) -> HalfDuplexUser:
    """Build a half-duplex user from a registered name.

    Uses the registry to resolve the user name to a constructor.

    Args:
        user_name: Registered user name (e.g., "user_simulator", "dummy_user").
        environment: The environment to extract user tools from.
        task: The task (used for user instructions).
        llm: LLM model name for the user simulator.
        llm_args: LLM arguments for the user simulator.
        persona_config: Persona configuration (verbosity, interrupt tendency).
        solo_mode: If True, validates that DummyUser is used appropriately.

    Returns:
        A fully constructed half-duplex user instance.

    Raises:
        AssertionError: If DummyUser is used without solo_mode.
    """
    UserConstructor = registry.get_user_constructor(user_name)

    try:
        user_tools = environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        user_tools = None

    # Validate DummyUser usage
    if issubclass(UserConstructor, DummyUser):
        assert solo_mode, "Dummy user can only be used with solo agent"

    user_kwargs = {
        "tools": user_tools,
        "instructions": str(task.user_scenario),
        "llm": llm,
        "llm_args": llm_args,
    }
    if issubclass(UserConstructor, UserSimulator):
        user_kwargs["persona_config"] = persona_config

    return UserConstructor(**user_kwargs)


# =============================================================================
# High-level build functions (use RunConfig)
# =============================================================================


def _derive_read_log_allowlist(task: Task) -> set:
    """Set of discoverable-tool names required by the task's golden trajectory.

    The banking_knowledge ``call_discoverable_agent_tool`` wrapper logs every
    call to the ``agent_discoverable_tools`` DB table, which is then hashed
    for the env-eval reward. For READ-only discoverable tools this means any
    extra validation call (e.g. an agent reading a balance "just in case")
    diverges the hash and zeros the reward — even though the KB explicitly
    encourages such reads.

    To keep the "agent must call this read to verify" assertion (tasks 046,
    085, etc.) while not punishing extra reads, we extract the set of tool
    names appearing in golden ``call_discoverable_agent_tool`` actions and
    pass it as an allowlist to the toolkit. Calls to writes are always
    logged; calls to reads are only logged when in this set.
    """
    allowlist: set = set()
    if task.evaluation_criteria is None:
        return allowlist
    for action in task.evaluation_criteria.actions or []:
        if action.name == "call_discoverable_agent_tool":
            name = (action.arguments or {}).get("agent_tool_name")
            if name:
                allowlist.add(name)
    return allowlist


def _build_env_kwargs(config: RunConfig, task: Task) -> dict:
    """Build env_kwargs from a RunConfig for the environment constructor.

    Extracts retrieval-related config (banking_knowledge domain) and includes
    the task reference needed for golden_retrieval policy.
    """
    env_kwargs: dict = {}
    retrieval_config = getattr(config, "retrieval_config", None)
    if retrieval_config is not None:
        env_kwargs["retrieval_variant"] = retrieval_config
        env_kwargs["task"] = task
        rk = dict(getattr(config, "retrieval_config_kwargs", None) or {})
        if rk:
            env_kwargs["retrieval_kwargs"] = rk
    if getattr(config, "domain", None) == "banking_knowledge":
        env_kwargs["read_log_allowlist"] = _derive_read_log_allowlist(task)
    return env_kwargs


def build_text_orchestrator(
    config: TextRunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    user_persona_config: Optional[PersonaConfig] = None,
) -> Orchestrator:
    """Build a half-duplex (text) orchestrator from a TextRunConfig.

    Args:
        config: Text run configuration.
        task: The task to run.
        seed: Per-trial seed. If None, uses config.seed.
        simulation_id: Unique simulation ID. If None, a UUID is generated.
        user_persona_config: Persona config for the user simulator.

    Returns:
        A fully constructed Orchestrator, ready for run_simulation().

    Example:
        config = TextRunConfig(domain="airline", agent="llm_agent")
        tasks = get_tasks("airline")
        orchestrator = build_text_orchestrator(config, tasks[0], seed=42)
        result = run_simulation(orchestrator)
    """
    if simulation_id is None:
        simulation_id = str(uuid.uuid4())
    if seed is None:
        seed = config.seed

    solo_mode = registry.get_agent_metadata(
        config.effective_agent, "solo_mode", default=False
    )
    domain = config.domain
    env_kwargs = _build_env_kwargs(config, task)

    environment = build_environment(domain, solo_mode=solo_mode, env_kwargs=env_kwargs)

    agent = build_agent(
        config.effective_agent,
        environment,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
        task=task,
        solo_mode=solo_mode,
    )

    user = build_user(
        config.effective_user,
        environment,
        task,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        persona_config=user_persona_config,
        solo_mode=solo_mode,
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.effective_max_steps,
        max_errors=config.max_errors,
        seed=seed,
        solo_mode=solo_mode,
        simulation_id=simulation_id,
        validate_communication=config.enforce_communication_protocol,
        timeout=config.timeout,
    )

    logger.debug(
        f"Built text orchestrator: domain={domain}, agent={config.effective_agent}, "
        f"user={config.effective_user}, task={task.id}"
    )

    return orchestrator


def build_orchestrator(
    config: RunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    user_persona_config: Optional[PersonaConfig] = None,
) -> Orchestrator:
    """Build a ready-to-run orchestrator from a RunConfig and task.

    Args:
        config: Run configuration.
        task: The task to run.
        seed: Per-trial seed. If None, uses config.seed.
        simulation_id: Unique simulation ID. If None, a UUID is generated.
        user_persona_config: Pre-computed persona config.

    Returns:
        A fully constructed Orchestrator.
    """
    return build_text_orchestrator(
        config,
        task,
        seed=seed,
        simulation_id=simulation_id,
        user_persona_config=user_persona_config,
    )
