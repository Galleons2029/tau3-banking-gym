"""Lightweight construction of the user simulator's initial system prompt."""

from __future__ import annotations

from pathlib import Path

from tau3.utils.utils import DATA_DIR

GLOBAL_USER_SIM_GUIDELINES_DIR = DATA_DIR / "tau3" / "user_simulator"
GLOBAL_USER_SIM_GUIDELINES_PATH = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines.md"
)
GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_tools.md"
)

SYSTEM_PROMPT = """
{global_user_sim_guidelines_with_persona}

<scenario>
{instructions}
</scenario>
""".strip()


def get_global_user_sim_guidelines(use_tools: bool = False) -> str:
    """Load the global guidelines used by the user simulator."""

    path: Path = (
        GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS
        if use_tools
        else GLOBAL_USER_SIM_GUIDELINES_PATH
    )
    return path.read_text()


def build_user_system_prompt(
    instructions: str | None,
    *,
    persona_guidelines: str | None = None,
    use_tools: bool = False,
) -> str:
    """Build the exact system prompt without importing the LLM runtime."""

    persona = persona_guidelines or ""
    if persona:
        persona = f"\n\n{persona}\n"
    guidelines = get_global_user_sim_guidelines(use_tools=use_tools).replace(
        "<PERSONA_GUIDELINES>", persona
    )
    return SYSTEM_PROMPT.format(
        global_user_sim_guidelines_with_persona=guidelines,
        instructions=instructions,
    )
