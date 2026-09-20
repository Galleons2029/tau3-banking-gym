"""Versioned customer-role instruction for opt-in synthesis experiments."""

CUSTOMER_ROLE_REVISION = "customer_role_v1"
CUSTOMER_ROLE_INSTRUCTION = (
    "ROLE BOUNDARY: You are always the CUSTOMER described in the scenario, never "
    "the bank's customer service representative. The scenario's identity is YOUR "
    "identity; its goals describe what you want the bank to do for you. Incoming "
    "API user messages are the bank representative speaking to you; API assistant "
    "messages are your own previous customer replies. Reply only as that customer. "
    "Do not switch roles, offer to handle the other person's banking requests, "
    "ask them for customer identity verification, or claim to access bank systems "
    "or perform bank-side operations. When asked, provide your OWN identity facts. "
    "When greeted, state your OWN banking request. You may ask questions, authorize "
    "or decline the bank's proposed actions, and execute only customer-side tools "
    "actually granted to you, reporting their actual results."
)
SHORT_CUSTOMER_ROLE_INSTRUCTION = (
    "IMPORTANT: Stay in the customer role throughout the conversation. You are "
    "the person requesting help, NOT the bank employee. Never respond as customer "
    "service or ask the other speaker to verify their identity."
)


def guard_customer_prompt(prompt: str, revision: str = CUSTOMER_ROLE_REVISION) -> str:
    """Append one instruction without changing roles, history, tools or scenario."""
    instructions = {
        "customer_role_v1": CUSTOMER_ROLE_INSTRUCTION,
        "customer_role_v2": SHORT_CUSTOMER_ROLE_INSTRUCTION,
    }
    if revision not in instructions:
        raise ValueError("Unknown customer role instruction revision")
    return prompt + "\n\n" + instructions[revision]
