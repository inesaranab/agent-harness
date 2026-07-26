# harness/supervisor.py
import asyncio
from typing import Literal

from dbos import DBOS
from pydantic import BaseModel

from harness.bus import emit
from harness.investigators import run_investigator
from harness.runtime import client, emit_step  # reuse both; also forces DBOS() first

MODEL = "gpt-5.6-luna"


# The PLAN is a first-class artifact: a structured object the supervisor emits,
# the inspector renders, the synthesis reads, and that survives a crash. It's
# untrusted model output crossing a trust boundary → a Pydantic model is right.
class PlanStep(BaseModel):
    id: str
    agent: Literal["billing", "technical", "sales"]
    objective: str


class Plan(BaseModel):
    steps: list[PlanStep]


@DBOS.step()
async def plan_step(task: str) -> list[dict]:
    # Structured output: the model must return something shaped like Plan.
    resp = await client.responses.parse(
        model=MODEL,
        instructions=(
            "Decompose a customer escalation into independent sub-tasks — one per "
            "area the message actually raises (billing / technical / sales). Only "
            "include relevant areas."
        ),
        input=task,
        text_format=Plan,
    )
    plan = resp.output_parsed
    return [step.model_dump() for step in plan.steps] if plan else []


@DBOS.step()
async def investigate_step(step: dict, workflow_id: str) -> dict:
    # started/completed are emitted INSIDE the step (like tool_step does).
    emit(
        {
            "type": "subagent.started",
            "workflowId": workflow_id,
            "stepId": step["id"],
            "agent": step["agent"],
            "objective": step["objective"],
        }
    )
    findings = await run_investigator(step["agent"], step["objective"])
    emit(
        {
            "type": "subagent.completed",
            "workflowId": workflow_id,
            "stepId": step["id"],
            "agent": step["agent"],
            "findings": findings,
        }
    )
    return {"agent": step["agent"], "findings": findings}


@DBOS.step()
async def synthesize_step(task: str, findings: list[dict]) -> str:
    joined = "\n\n".join(f"[{f['agent']}] {f['findings']}" for f in findings) or "(none)"
    resp = await client.responses.create(
        model=MODEL,
        instructions=(
            "You are a support lead. Using your investigators' findings, write ONE "
            "clear, friendly reply to the customer that addresses every point they "
            "raised. If an area's investigation is missing, acknowledge it briefly "
            "and say you'll follow up."
        ),
        input=f"Customer escalation:\n{task}\n\nInvestigator findings:\n{joined}",
    )
    return resp.output_text


# THE SUPERVISOR. Plan → dispatch sub-agents in parallel → fan in → synthesize.
# Unlike a handoff, the supervisor keeps control the whole time.
@DBOS.workflow()
async def supervisor_workflow(task: str) -> str:
    workflow_id = DBOS.workflow_id or "unknown"
    await emit_step({"type": "workflow.started", "workflowId": workflow_id, "input": task})

    # PLAN
    steps = await plan_step(task)
    await emit_step({"type": "plan.created", "workflowId": workflow_id, "steps": steps})

    # DISPATCH — every sub-agent runs in parallel, each in its own context window.
    results = await asyncio.gather(
        *(investigate_step(step, workflow_id) for step in steps),
        return_exceptions=True,
    )

    # FAN-IN — keep successes, record failures, and keep going (degrade).
    findings: list[dict] = []
    for step, result in zip(steps, results):
        if isinstance(result, BaseException):
            await emit_step(
                {
                    "type": "subagent.failed",
                    "workflowId": workflow_id,
                    "stepId": step["id"],
                    "agent": step["agent"],
                    "error": str(result),
                }
            )
        else:
            findings.append(result)

    # SYNTHESIZE
    reply = await synthesize_step(task, findings)
    await emit_step({"type": "model.completed", "workflowId": workflow_id, "text": reply})
    await emit_step({"type": "workflow.completed", "workflowId": workflow_id, "output": reply})
    return reply
