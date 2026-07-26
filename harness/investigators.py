import json
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam

from config import settings
from harness.tools import TOOL_SCHEMAS, get_charges, search_knowledge_base

client = AsyncOpenAI(api_key=settings.openai_api_key)

MODEL = "gpt-5.6-luna"
MAX_STEPS = 5

# Reuse the shared schema
_SEARCH_KB = next(s for s in TOOL_SCHEMAS if s["name"] == "searchKnowledgeBase")

# getCharges has NO shared schema on purpose — the main agent reaches charges via
# the sandbox (runCode), not a direct tool. Defining it here keeps it inside the
# read-only investigator subsystem instead of widening the main agent's surface.
_GET_CHARGES: FunctionToolParam = {
    "type": "function",
    "strict": False,
    "name": "getCharges",
    "description": "Look up a customer's charges.",
    "parameters": {
        "type": "object",
        "properties": {"customerId": {"type": "string"}},
        "required": ["customerId"],
    },
}

# name -> how to actually run it. Local too: run_tool intentionally can't reach
# getCharges, and these are read-only so a re-run on recovery is safe.
_TOOL_FNS = {
    "getCharges": lambda a: get_charges(
        a["customerId"]
    ),  # call.arguments == '{"customerId": "cus_88121"}' #noqa
    "searchKnowledgeBase": lambda a: search_knowledge_base(a["query"]),
}


@dataclass(frozen=True)
class Investigator:
    system_prompt: str
    tools: tuple[FunctionToolParam, ...]


INVESTIGATORS: dict[str, Investigator] = {
    "billing": Investigator(
        system_prompt="You are a billing investigator. Use getCharges to find duplicate or "
        "erroneous charges. Report the charge ids, the amount, and the refund "
        "you'd recommend — concisely.",
        tools=(_GET_CHARGES,),
    ),
    "technical": Investigator(
        system_prompt="You are a technical investigator. Use searchKnowledgeBase to find known "
        "bugs and workarounds. Report the issue, any ticket, and the workaround — "
        "concisely.",
        tools=(_SEARCH_KB,),
    ),
    "sales": Investigator(
        system_prompt="You are a sales investigator. Use searchKnowledgeBase for pricing "
        "guidance, then state the relevant numbers and next step — concisely.",
        tools=(_SEARCH_KB,),
    ),
}


async def run_investigator(agent: str, objective: str):
    """Run one investigator over its objective in its OWN context. Returns findings."""
    investigator = INVESTIGATORS.get(agent)
    if investigator is None:
        raise ValueError(f"unknown investigator: {agent}")

    # Turn 1 is a plain inference: system via `instructions`, objective via `input`.
    input_items: str | list = objective
    resp = None

    for _ in range(MAX_STEPS):
        resp = await client.responses.create(
            model=MODEL,
            instructions=investigator.system_prompt,
            input=input_items,
            tools=investigator.tools,
        )
        calls = [item for item in resp.output if item.type == "function_call"]
        if not calls:
            return resp.output_text

        # A tool fired -> upgrade to the list form so we can feed the result back.
        if isinstance(input_items, str):
            input_items = [{"role": "user", "content": input_items}]
        input_items += [item.model_dump(exclude={"status"}) for item in resp.output]
        for call in calls:
            result = _TOOL_FNS[call.name](json.loads(call.arguments))
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )
    return resp.output_text if resp else ""
