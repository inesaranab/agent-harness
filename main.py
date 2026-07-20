import asyncio

from dbos import DBOS

from harness.bus import subscribe
from harness.db import ensure_schema
from harness.runtime import agent_workflow
from harness.system_prompt import SAMPLE_TASK


async def _entrypoint() -> None:
    subscribe(lambda e: print(e.get("type")))  # console view of the event stream
    handle = await DBOS.start_workflow_async(agent_workflow, SAMPLE_TASK)
    await handle.get_result()


if __name__ == "__main__":
    ensure_schema()
    DBOS.launch()
    asyncio.run(_entrypoint())
