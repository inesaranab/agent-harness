# harness/runtime.py
import json

from dbos import DBOS, DBOSConfig
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)

from config import settings
from harness.bus import emit
from harness.memory import (
    KEEP_CONTEXT_TOKENS,
    MAX_CONTEXT_TOKENS,
    build_context,
    estimate_tokens,
    summarize,
)
from harness.tools import TOOL_SCHEMAS, run_tool

DBOS(config=DBOSConfig(name="ines-harness", system_database_url=settings.database_url))

client = AsyncOpenAI(api_key=settings.openai_api_key)


MAX_STEPS = 30


@DBOS.step()
async def emit_step(event: dict) -> None:
    emit(event)


@DBOS.step()
async def summarize_step(old_turns: list, prior_summary: str) -> str:
    return await summarize(old_turns, prior_summary)


@DBOS.step()
async def model_turn(workflow_id: str, context: list) -> dict:
    tool_calls = []

    # One model turn over the HYDRATED context (not the whole history).
    async with client.responses.stream(
        model="gpt-5.6-luna",
        input=context,
        tools=TOOL_SCHEMAS,
    ) as stream:
        async for event in stream:
            match event:
                case ResponseTextDeltaEvent():
                    emit({"type": "model.delta", "workflowId": workflow_id, "text": event.delta})
                case ResponseOutputItemDoneEvent() if isinstance(
                    event.item, ResponseFunctionToolCall
                ):
                    emit(
                        {
                            "type": "tool.requested",
                            "workflowId": workflow_id,
                            "toolCallId": event.item.call_id,
                            "name": event.item.name,
                            "args": event.item.arguments,
                        }
                    )
                    tool_calls.append(
                        {
                            "name": event.item.name,
                            "arguments": event.item.arguments,
                            "call_id": event.item.call_id,
                        }
                    )
                case ResponseErrorEvent():
                    emit(
                        {
                            "type": "workflow.failed",
                            "workflowId": workflow_id,
                            "error": event.message,
                        }
                    )
                    raise RuntimeError(event.message)
        final = await stream.get_final_response()

    return {
        "output": [
            item.model_dump(exclude={"status", "parsed_arguments"}) for item in final.output
        ],
        "output_text": final.output_text,
        "tool_calls": tool_calls,
    }


@DBOS.step()
async def tool_step(workflow_id: str, call: dict) -> dict:
    args = json.loads(call["arguments"])
    result = await run_tool(call["name"], args)
    emit(
        {
            "type": "tool.completed",
            "workflowId": workflow_id,
            "toolCallId": call["call_id"],
            "name": call["name"],
            "result": result,
        }
    )
    return result


@DBOS.workflow()
async def agent_workflow(user_input: str) -> str:
    workflow_id = DBOS.workflow_id
    await emit_step({"type": "workflow.started", "workflowId": workflow_id, "input": user_input})

    # Conversation as a list of TURNS to compact at clean boundaries.
    turns: list = []
    summary = ""

    step = 0
    while step < MAX_STEPS:
        # 1. Compact: while the recent window is over budget, peel the oldest
        #    turns into the running summary (keeping at least the last turn).
        if estimate_tokens([m for turn in turns for m in turn]) > MAX_CONTEXT_TOKENS:
            old: list = []
            while (
                len(turns) > 1
                and estimate_tokens([m for turn in turns for m in turn]) > KEEP_CONTEXT_TOKENS
            ):
                old.append(turns.pop(0))
            if old:
                summary = await summarize_step(old, summary)
                context_tokens = estimate_tokens(build_context(user_input, summary, turns))
                await emit_step(
                    {
                        "type": "memory.compacted",
                        "workflowId": workflow_id,
                        "summarizedTurns": len(old),
                        "contextTokens": context_tokens,
                        "summary": summary,
                    }
                )

        # 2 + 3. Hydrate the context and run one turn over it.
        context = build_context(user_input, summary, turns)
        turn = await model_turn(workflow_id, context)  # ty: ignore[invalid-argument-type]

        turn_messages: list = list(turn["output"])

        # No tool calls means the model answered. We're done.
        if not turn["tool_calls"]:
            await emit_step(
                {"type": "model.completed", "workflowId": workflow_id, "text": turn["output_text"]}
            )
            await emit_step(
                {
                    "type": "workflow.completed",
                    "workflowId": workflow_id,
                    "output": turn["output_text"],
                }
            )
            return turn["output_text"]

        # Run each requested tool, feed the result into THIS turn's messages.
        for call in turn["tool_calls"]:
            result = await tool_step(workflow_id, call)  # ty: ignore[invalid-argument-type]
            turn_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(result),
                }
            )

        turns.append(turn_messages)
        step += 1

    await emit_step(
        {
            "type": "workflow.failed",
            "workflowId": workflow_id,
            "error": f"Hit the {MAX_STEPS}-step limit.",
        }
    )
    return ""
