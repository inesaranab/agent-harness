# Agent Harness

An event-driven agent runtime: a durable ReAct loop with memory compaction and
multi-agent **handoffs**, plus a live inspector that renders every event the
harness emits.


## Demo

Triage classifies each work item, then hands a refund off to the billing
specialist. The full event stream — streamed model tokens, tool calls, memory
compaction, and the handoff itself — renders live in the inspector on the right.


https://github.com/user-attachments/assets/87b7c105-ddb6-4ca3-81af-6efeb6446b15

 Toggle Supervised and the harness plans first: it decomposes the escalation into independent sub-tasks, dispatches a specialist investigator for each in parallel, then synthesizes their findings into one reply. The plan and every sub-agent — running, done, or failed — render live in the inspector on the right.

 
https://github.com/user-attachments/assets/8897b7a8-c21c-41c0-8024-d57ccf834ccd







