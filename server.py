import asyncio
import logging
from contextlib import asynccontextmanager

from dbos import DBOS
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from sqlalchemy import text

from harness.bus import emit, history, subscribe
from harness.code_mode import call
from harness.db import db_client, ensure_schema
from harness.runtime import agent_workflow
from harness.supervisor import supervisor_workflow

logger = logging.getLogger("harness.rpc")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_schema()
    DBOS.launch()
    yield
    DBOS.destroy()


app = FastAPI(lifespan=lifespan)


# Keep references so background runs aren't garbage-collected mid-execution.
_running_tasks: set[asyncio.Task] = set()


async def run_task(task: str, mode: str = "default") -> None:
    workflow_id = ""
    try:
        if mode == "supervised":
            handle = await DBOS.start_workflow_async(supervisor_workflow, task)
        else:
            handle = await DBOS.start_workflow_async(agent_workflow, task)
        workflow_id = handle.workflow_id
        await handle.get_result()
    except Exception as e:
        emit({"type": "workflow.failed", "workflowId": workflow_id, "error": str(e)})


@app.post("/rpc/tool")
async def rpc_tool(req: Request):
    try:
        body = await req.json()
        return {"result": call(body["token"], body["name"], body["args"])}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception:
        # Return generic message (no internals)
        logger.exception("rpc_tool failed")
        return {"error": "internal error"}


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
                    mode = msg.get("mode", "default")
                    # Run the agent in the background (agent -> queue), keeping a
                    # reference so the task isn't garbage-collected mid-run.
                    t = asyncio.create_task(run_task(task, mode))
                    _running_tasks.add(t)
                    t.add_done_callback(_running_tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        # 6. Clean up when the browser leaves.
        forward_task.cancel()
        unsub()


@app.post("/reset")
def reset():
    with db_client.begin() as conn:
        conn.execute(text("TRUNCATE TABLE event_log RESTART IDENTITY"))
    return {"ok": True}
