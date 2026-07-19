import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from harness.bus import emit, history, subscribe
from harness.runtime import run_agent

app = FastAPI()


async def run_task(task: str) -> None:
    # Wrap the run so an uncaught error surfaces as an event instead of vanishing.
    try:
        await run_agent(task, emit)
    except Exception as e:
        emit({"type": "workflow.failed", "workflowId": "", "error": str(e)})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    # 1. Accept the browser's connection.
    await websocket.accept()

    # 2. Bridge the sync bus to the async socket
    queue: asyncio.Queue = asyncio.Queue()
    unsub = subscribe(lambda e: queue.put_nowait(e))

    # 3. Replay the whole timeline from Postgres
    for event in history():
        await websocket.send_json(event)

    # 4. Push queued events to the browser, forever
    async def forward():
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except Exception:
            pass

    forward_task = asyncio.create_task(forward())

    # 5. Receive commands from the browser (browser -> agent).
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "submit_task":
                task = msg.get("input")
                if task:
                    # Run the agent in the background (agent -> queue)
                    asyncio.create_task(run_task(task))
    except WebSocketDisconnect:
        pass
    finally:
        # 6. Clean up when the browser leaves.
        forward_task.cancel()
        unsub()
