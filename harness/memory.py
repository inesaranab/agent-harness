import json
from functools import cache

import tiktoken
from openai import AsyncOpenAI

from config import settings

# Small on purpose so a short task triggers compaction.
MAX_CONTEXT_TOKENS = 500
KEEP_CONTEXT_TOKENS = 200

client = AsyncOpenAI(api_key=settings.openai_api_key)


@cache
def _encoder() -> tiktoken.Encoding:
    # Built once, reused forever (lazy).
    return tiktoken.get_encoding("o200k_base")


def _as_text(message: object) -> str:
    # Messages are mixed dicts (role/content, function_call_output, ...)
    return message if isinstance(message, str) else json.dumps(message)


# ── estimate: how many tokens a list of messages costs ──────────────
def estimate_tokens(messages: list) -> int:
    text = "\n".join(_as_text(m) for m in messages)
    # encode_ordinary never raises on text that looks like a special token
    return len(_encoder().encode_ordinary(text))


# ── CONTEXT: what the model actually sees THIS turn, assembled fresh ─
def build_context(task: str, summary: str, turns: list, system_prompt: str) -> list:
    context: list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},  # the goal is pinned, never summarized away
    ]
    if summary:
        context.append({"role": "system", "content": f"Summary of earlier work so far:\n{summary}"})
    for turn in turns:
        context.extend(turn)  # the most recent turns, verbatim
    return context


# ── STATE: compress old turns into the running summary (an LLM call) ─
async def summarize(old_turns: list, prior_summary: str) -> str:
    transcript = "\n".join(_as_text(m) for turn in old_turns for m in turn)[:6000]
    resp = await client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "system",
                "content": (
                    "You compress an agent's work log into a short running summary. "
                    "Preserve concrete facts: item ids, categories, draft ids, amounts, "
                    "and what was already sent. Be terse."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Prior summary:\n{prior_summary or '(none)'}\n\n"
                    f"Fold in this newer work:\n{transcript}\n\n"
                    "Return the updated summary."
                ),
            },
        ],
    )
    return resp.output_text
