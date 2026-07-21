# harness/sandbox.py
import asyncio
from textwrap import indent

from e2b_code_interpreter import AsyncSandbox

from config import settings
from harness.code_mode import issue_token, revoke
from harness.tools import READ_TOOLS

RPC_TIMEOUT_S = 10  # per tool call from inside the sandbox
RUN_TIMEOUT_S = 30  # for the whole program


_SHIM = """import json, urllib.request
_RPC, _TOKEN = {rpc!r}, {token!r}
class _Tools:
    def __getattr__(self, name):
        def _call(*args):
            body = json.dumps({{"token": _TOKEN, "name": name, "args": list(args)}}).encode()
            req = urllib.request.Request(_RPC, body, {{"Content-Type": "application/json"}})
            resp = json.loads(urllib.request.urlopen(req, timeout={timeout}).read())
            if "error" in resp:
                raise RuntimeError(resp["error"])
            return resp["result"]
        return _call
tools = _Tools()
"""


async def run_in_sandbox(code: str) -> dict:
    token = issue_token(READ_TOOLS)
    shim = _SHIM.format(rpc=settings.rpc_base_url + "/rpc/tool", token=token, timeout=RPC_TIMEOUT_S)
    # Wrap so a top-level `return` is valid, then evaluate its result.
    wrapped = "def _main():\n" + indent(code, "    ") + "\n_result = _main()\n_result"
    try:
        async with await AsyncSandbox.create(api_key=settings.e2b_api_key) as sbx:
            execution = await asyncio.wait_for(
                sbx.run_code(shim + "\n" + wrapped), timeout=RUN_TIMEOUT_S
            )
        if execution.error:
            return {"ok": False, "error": str(execution.error), "logs": execution.logs.stdout}
        return {"ok": True, "result": execution.text, "logs": execution.logs.stdout}
    except TimeoutError:
        return {"ok": False, "error": f"sandbox timed out after {RUN_TIMEOUT_S}s", "logs": []}
    finally:
        revoke(token)
