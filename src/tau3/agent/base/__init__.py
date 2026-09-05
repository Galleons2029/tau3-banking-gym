"""
Base agent components.

This module exports the foundational building blocks for creating
half-duplex (turn-based) conversation participants.
"""

# LLM configuration mixin
from tau3.agent.base.llm_config import LLMConfigMixin

# Protocol base class
from tau3.agent.base.participant import HalfDuplexParticipant

# =============================================================================
# DEPRECATION ALIASES
# =============================================================================

# Deprecated alias for backward compatibility
BaseConversationParticipant = HalfDuplexParticipant


__all__ = [
    # Protocol base class
    "HalfDuplexParticipant",
    # LLM configuration
    "LLMConfigMixin",
    # Deprecated alias (kept for backward compatibility)
    "BaseConversationParticipant",
]
