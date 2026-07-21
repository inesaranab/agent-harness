# harness/tools.py

from collections.abc import Callable

from openai.types.responses import FunctionToolParam

# dictionaries for testing
KNOWLEDGE_BASE = {
    "billing": "Double charges are usually a duplicate authorization that drops off in 3–5 days. If it already settled, refund immediately.",  # noqa: E501
    "refund": "Refunds post in 5–10 business days. Pro accounts can be expedited.",
    "export": "The Safari export failure is a known bug (TICKET-4412). Workaround: use Chrome or the CSV export.",  # noqa: E501
    "pricing": "Team plans are $20/seat/mo with a volume discount at 25+ seats. For 50+ seats, send the pricing PDF.",  # noqa: E501
}
CHARGES = {
    "cus_88121": [
        {"id": "ch_001", "amount": 4900, "date": "2026-05-01", "description": "Pro plan — monthly"},
        {"id": "ch_002", "amount": 4900, "date": "2026-05-01", "description": "Pro plan — monthly"},
        {"id": "ch_003", "amount": 1500, "date": "2026-04-18", "description": "Extra seats"},
    ],
}


# ── the functions (what actually runs) ──────────────────────────────
def search_knowledge_base(query: str) -> dict:
    hits = [a for key, a in KNOWLEDGE_BASE.items() if key in query.lower()]
    return {"articles": hits or ["No exact match — use your judgment."]}


def classify_item(itemId: str, category: str) -> dict:
    return {"ok": True, "itemId": itemId, "category": category}


def draft_reply(itemId: str, message: str) -> dict:
    return {"ok": True, "draftId": f"draft-{itemId}"}


def send_reply(itemId: str, draftId: str) -> dict:
    # DANGEROUS: irreversible side effect, zero confirmation. Fixed later.
    return {"sent": True, "itemId": itemId, "draftId": draftId}


def get_charges(customer_id: str) -> list:
    return CHARGES.get(customer_id, [])


# ── the schemas (what the model reads to decide what to call) ────────
TOOL_SCHEMAS: list[FunctionToolParam] = [
    {
        "type": "function",
        "strict": False,
        "name": "runCode",
        "description": "Run a Python program (a function body) to fetch and analyze data. Inside you can call tools.getCharges(customer_id) and tools.searchKnowledgeBase(query). Use `return` to return your result (any JSON value).",  # noqa: E501
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "type": "function",
        "strict": False,
        "name": "searchKnowledgeBase",
        "description": "Search the support knowledge base for relevant articles.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "strict": False,
        "name": "classifyItem",
        "description": "Classify a work item into a category.",
        "parameters": {
            "type": "object",
            "properties": {
                "itemId": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["billing", "technical", "sales", "other"],
                },
            },
            "required": ["itemId", "category"],
        },
    },
    {
        "type": "function",
        "strict": False,
        "name": "draftReply",
        "description": "Write a draft reply for a work item. Does not send anything.",
        "parameters": {
            "type": "object",
            "properties": {
                "itemId": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["itemId", "message"],
        },
    },
    {
        "type": "function",
        "strict": False,
        "name": "sendReply",
        "description": "Send the drafted reply to the customer. This really emails them.",
        "parameters": {
            "type": "object",
            "properties": {
                "itemId": {"type": "string"},
                "draftId": {"type": "string"},
            },
            "required": ["itemId", "draftId"],
        },
    },
]


# ── the registry (name → function) ──────────────────────────────────
TOOLS = {
    "searchKnowledgeBase": search_knowledge_base,
    "classifyItem": classify_item,
    "draftReply": draft_reply,
    "sendReply": send_reply,
}

# for code mode
READ_TOOLS: dict[str, Callable] = {
    "getCharges": get_charges,
    "searchKnowledgeBase": search_knowledge_base,
}


# ── the single execution entrypoint (his runTool) ───────────────────
async def run_tool(name: str, args: dict) -> dict:
    if name == "runCode":
        from harness.sandbox import run_in_sandbox  # lazy import: avoids a cycle

        return await run_in_sandbox(args["code"])
    return TOOLS[name](**args)
