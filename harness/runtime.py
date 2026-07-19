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
from harness.system_prompt import SYSTEM_PROMPT
from harness.tools import TOOL_SCHEMAS, run_tool

DBOS(config=DBOSConfig(name="ines-harness", system_database_url=settings.database_url))

client = AsyncOpenAI(api_key=settings.openai_api_key)

# Loop guard
MAX_STEPS = 10


@DBOS.step()
async def emit_step(event: dict) -> None:
    emit(event)


@DBOS.step()
async def model_turn(workflow_id: str, messages: list) -> dict:
    tool_calls = []

    # Stream — we match on the SDK's event CLASSES, not strings.
    async with client.responses.stream(
        model="gpt-5.6-luna",
        input=messages,
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
                    # Stop the turn instead of falling into get_final_response()
                    # on an already-errored stream.
                    raise RuntimeError(event.message)
        final = await stream.get_final_response()

    # Return only serializable data — DBOS checkpoints this to Postgres.
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
    result = run_tool(call["name"], args)
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

    # STATE
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # THE LOOP.
    step = 0
    while step < MAX_STEPS:
        turn = await model_turn(workflow_id, messages)  # ty: ignore[invalid-argument-type]
        # Append the model's output to history so the next turn sees it.
        messages += turn["output"]

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

        # Run each requested tool with NO mediation, feed the result back.
        for call in turn["tool_calls"]:
            result = await tool_step(workflow_id, call)  # ty: ignore[invalid-argument-type]
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(result),
                }
            )

        step += 1

    await emit_step(
        {
            "type": "workflow.failed",
            "workflowId": workflow_id,
            "error": f"Hit the {MAX_STEPS}-step limit.",
        }
    )
    return ""
