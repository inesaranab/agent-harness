import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from harness.bus import emit, history, subscribe
from harness.runtime import run_agent

app = FastAPI()


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
        while True:
            event = await queue.get()
            await websocket.send_json(event)

    forward_task = asyncio.create_task(forward())

    # 5. Receive commands from the browser (browser -> agent).
    try:
        while True:
            msg = await websocket.receive_json()
            if msg["type"] == "submit_task":
                # Run the agent in the background; it reports via emit -> queue.
                asyncio.create_task(run_agent(msg["input"], emit))
    except WebSocketDisconnect:
        pass
    finally:
        # 6. Clean up when the browser leaves.
        forward_task.cancel()
        unsub()
