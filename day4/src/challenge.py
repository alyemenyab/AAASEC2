"""
DAY 4 — CHALLENGE template (heavily scaffolded — read top to bottom).

READ FIRST:  ../04-challenge.md

This file is a fill-in-the-blanks walkthrough. Every ▢ marks a spot
where you write a few lines; everything else is done and commented.
If you completed 00-03, each blank is something you have already
written once today. Total new code: roughly 25 lines.

The shape of what you're building:

    your prompt
        │
        ▼
    Deep Agent ──(tool)──► your authenticated MCP server   [information]
        │
        └──(backend)─────► execute on your machine          [computation]
                                    │
                              LangSmith trace               [visibility]

Run order:
    terminal 1:  uv run python src/secure_mcp.py
    terminal 2:  uv run python src/challenge.py
    browser   :  smith.langchain.com -> project aaasec2-day4 -> your run
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

from deepagents import create_deep_agent

# Reuse what you already built. If your names differ, fix the import,
# not your files.
try:
    from .shell_agent import SYSTEM_PROMPT, llm, make_backend
except ImportError:  # pragma: no cover — running as a loose script
    from shell_agent import SYSTEM_PROMPT, llm, make_backend

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://localhost:8002/mcp")
ADMIN_TOKEN = os.getenv("MCP_ADMIN_TOKEN", "admin-secret-token")


# ════════════════════════════════════════════════════════════════
# STEP 1 — your own protected capability
# ════════════════════════════════════════════════════════════════
# The mission used OUR get_internal_report. The challenge wants YOUR
# data behind YOUR protected tool.
#
# ✅ 1a. DONE — src/secure_mcp.py now also serves get_lab_inventory:
#        drone-lab parts with quantity, unit_cost and reorder_level,
#        behind the same @mcp.tool(auth=require_scopes("read:internal"))
#        decorator as get_internal_report. Same pattern, my data.
#
# ✅ 1b. Its tool name:

MY_TOOL_NAME = "get_lab_inventory"


# ════════════════════════════════════════════════════════════════
# STEP 2 — the tool wrapper (given; identical to mission.py's)
# ════════════════════════════════════════════════════════════════
# Sync outside, async inside, asyncio.run as the bridge. Explained
# line by line in src/mission.py and src/check_auth.py — this is the
# third time you've seen it, which is the point.

def fetch_my_data() -> str:
    """Fetch my protected dataset from my secure MCP server."""

    async def _call() -> str:
        async with Client(MCP_URL, auth=BearerAuth(token=ADMIN_TOKEN)) as c:
            result = await c.call_tool(MY_TOOL_NAME, {})
            return json.dumps(result.data)

    return asyncio.run(_call())


# ════════════════════════════════════════════════════════════════
# STEP 3 — your task prompt
# ════════════════════════════════════════════════════════════════
# ▢ Write a MISSION for your data. Keep the mission.py rhythm:
#     fetch -> write a program -> execute it -> report what it printed.
#   The "report exactly what the program printed" clause is what lets
#   you catch the model summarizing instead of computing.

MISSION = (
    "1. Call fetch_my_data to get the lab inventory. "
    "2. Write restock.py that, for every item, computes stock value "
    "(quantity * unit_cost) and the shortfall against reorder_level "
    "(reorder_level - quantity, floored at 0); then prints the total "
    "inventory value, a table of every item that is at or below its "
    "reorder level with the cost of restocking it back to that level, "
    "and the total cost of that restock. Round money to 2 decimals. "
    "3. Execute it with python. "
    "4. Report exactly what the program printed, plus one insight."
)


# ════════════════════════════════════════════════════════════════
# STEP 4 — assemble and run (given)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    backend, cleanup = make_backend()
    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[fetch_my_data],
            backend=backend,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": MISSION}]})
        print(result["messages"][-1].content)
    finally:
        cleanup()


# ════════════════════════════════════════════════════════════════
# STEP 5 — evidence (nothing to code)
# ════════════════════════════════════════════════════════════════
# ✅ 5a. Auth, both directions — `uv run python src/check_auth.py` with
#        the server up. Recorded output (get_lab_inventory substituted
#        into the protected rows):
#
#          ❌ no token, public tool          -> 401, rejected at the door
#          ❌ wrong token, public tool       -> 401
#          ✅ student token, public tool     -> 2026-08-12T...
#          ❌ student token, PROTECTED tool  -> "Unknown tool"
#          ✅ admin token, PROTECTED tool    -> {"lab": "AAASEC2 aerial...
#
#        Row 4 is the interesting one: not "forbidden" but Unknown tool.
#        401 is authentication failing; Unknown tool is authorization
#        working — FastMCP filters unauthorized tools out of discovery,
#        so an unprivileged caller cannot even learn the tool exists,
#        let alone probe the shape of the data behind it.
#
# ▢ 5b. LangSmith trace: run this file with LANGSMITH_TRACING=true and a
#       key in .env, then open smith.langchain.com -> aaasec2-day4 ->
#       the newest run. Read it in order: fetch_my_data (the MCP call)
#       -> write_file (restock.py) -> execute (python restock.py) ->
#       the printed table coming back into the model. Paste the run URL
#       here for the deliverable:
#
#         trace: <paste your run URL>
#
#       Check while you are there that the numbers in the final answer
#       are the numbers the trace shows restock.py printing. If they
#       drift, the model summarized instead of computing.
#
# ✅ 5c. The adversarial poke. `env` comes back nearly empty — PATH and
#        PWD, nothing else — because the shell was given an explicit env
#        instead of inheriting mine, so there are no API keys in there to
#        leak. The file read is the uncomfortable half. `~/.ssh/...` fails,
#        but for an accidental reason: HOME was stripped too, so the shell
#        cannot expand `~`. Spell the path out — /root/.ssh/id_ed25519.pub,
#        /etc/passwd, anything my user can read — and it comes straight
#        back; `whoami` answers with my own account. root_dir and
#        virtual_mode confine the agent's FILE TOOLS; they do not confine
#        a shell command, and the shell is the one with my permissions.
#
#        One sentence: the poke got whatever my own user account can
#        read, because the only thing pretending otherwise was a system
#        prompt — and a prompt is a request, not a boundary; what would
#        have stopped it is running `execute` somewhere that simply does
#        not contain my home directory (a sandbox: 05-extra-sandbox.md),
#        because the boundary belongs in the infrastructure.
