# harness/runtime.py

import json
from collections.abc import Callable

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from config import settings
from harness.system_prompt import SYSTEM_PROMPT
from harness.tools import TOOL_SCHEMAS, run_tool

client = AsyncOpenAI(api_key=settings.openai_api_key)

# Loop guard
MAX_STEPS = 10


async def run_agent(user_input: str, emit: Callable[[dict], None]) -> None:
    emit({"type": "workflow.started", "input": user_input})

    # STATE
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # THE LOOP.
    step = 0
    while step < MAX_STEPS:
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
                        emit({"type": "model.delta", "text": event.delta})
                    case ResponseOutputItemDoneEvent() if isinstance(
                        event.item, ResponseFunctionToolCall
                    ):
                        emit(
                            {
                                "type": "tool.requested",
                                "name": event.item.name,
                                "args": event.item.arguments,
                            }
                        )
                        tool_calls.append(event.item)
                    case ResponseErrorEvent():
                        emit({"type": "workflow.failed", "error": event.message})
                        return

            final = await stream.get_final_response()

        # Append the model's output to history so the next turn sees it.
        messages += [
            item.model_dump(exclude={"status", "parsed_arguments"}) for item in final.output
        ]

        # No tool calls means the model answered. We're done.
        if not tool_calls:
            emit({"type": "model.completed", "text": final.output_text})
            emit({"type": "workflow.completed", "output": final.output_text})
            return

        # Run each requested tool with NO mediation, feed the result back.
        for call in tool_calls:
            args = json.loads(call.arguments)
            result = run_tool(call.name, args)
            emit({"type": "tool.completed", "name": call.name, "result": result})
            out: FunctionCallOutput = {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            }
            messages.append(out)

        step += 1

    emit({"type": "workflow.failed", "error": f"Hit the {MAX_STEPS}-step limit."})
