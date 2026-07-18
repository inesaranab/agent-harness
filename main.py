from harness.runtime import run_agent
from harness.system_prompt import SAMPLE_TASK

if __name__ == "__main__":
    run_agent(SAMPLE_TASK, emit=print)
