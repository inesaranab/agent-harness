from dataclasses import dataclass

from openai.types.responses import FunctionToolParam

from harness.tools import TOOL_SCHEMAS

# schema-name to schema so an agent con pick its tools by name (its own subset of tools)
_SCHEMA_BY_NAME = {s["name"]: s for s in TOOL_SCHEMAS}


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str
    tools: tuple[str, ...]


def tool_schema_for(agent: Agent) -> list[FunctionToolParam]:
    """Select the agent tools subset by name"""
    own = [_SCHEMA_BY_NAME[name] for name in agent.tools]
    return own


TRIAGE = Agent(
    name="triage",
    system_prompt=(
        "You are a support triage agent. Classify each work item with "
        "classifyItem, then route it to the specialist that can resolve it. "
        "When every item is resolved, briefly summarize what was done and stop."
    ),
    tools=("classifyItem", "runCode", "draftReply", "sendReply", "handoff"),
)

BILLING = Agent(
    name="billing",
    system_prompt=(
        "You are a billing specialist. Use runCode to fetch and analyze charges "
        "— never do the arithmetic yourself. Draft a reply with draftReply, then "
        "send it with sendReply."
    ),
    tools=("runCode", "draftReply", "sendReply", "issueRefund"),
)


# register the agents to be able to export them
REGISTRY = {"triage": TRIAGE, "billing": BILLING}

# entrypoint agent
START_AGENT = TRIAGE
