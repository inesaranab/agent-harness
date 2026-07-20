import asyncio
from contextlib import asynccontextmanager

from dbos import DBOS
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from harness.bus import emit, history, subscribe
from harness.db import ensure_schema
from harness.runtime import agent_workflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_schema()
    DBOS.launch()
    yield
    DBOS.destroy()


app = FastAPI(lifespan=lifespan)


# Keep references so background runs aren't garbage-collected mid-execution.
_running_tasks: set[asyncio.Task] = set()


async def run_task(task: str) -> None:
    workflow_id = ""
    try:
        handle = await DBOS.start_workflow_async(agent_workflow, task)
        workflow_id = handle.workflow_id
        await handle.get_result()
    except Exception as e:
        emit({"type": "workflow.failed", "workflowId": workflow_id, "error": str(e)})


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
                    # Run the agent in the background (agent -> queue), keeping a
                    # reference so the task isn't garbage-collected mid-run.
                    t = asyncio.create_task(run_task(task))
                    _running_tasks.add(t)
                    t.add_done_callback(_running_tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        # 6. Clean up when the browser leaves.
        forward_task.cancel()
        unsub()
