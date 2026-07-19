import asyncio

from dbos import DBOS

from harness.bus import subscribe
from harness.db import ensure_schema
from harness.runtime import agent_workflow
from harness.system_prompt import SAMPLE_TASK

if __name__ == "__main__":
    ensure_schema()
    DBOS.launch()
    subscribe(lambda e: print(e.get("type")))  # console view of the event stream
    asyncio.run(agent_workflow(SAMPLE_TASK))
