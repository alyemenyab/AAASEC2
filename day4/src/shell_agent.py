"""
DAY 4 — Deep Agent with shell access.

READ FIRST:  ../00-deep-agent-shell.md

Yesterday's agent could read and write files. Today's agent has exactly
one more tool:

    execute        <- a real shell. On YOUR machine.

Same create_deep_agent call, different backend. That is the whole
difference, and it is not a small one.

⚠  LocalShellBackend has NO isolation. Commands run as your user, on
   your host, with your permissions. Two mitigations are set below:

     root_dir + virtual_mode  — the FILE tools are confined to day4/work/
     env={"PATH": ...}        — the shell does not inherit your
                                environment, so OPENAI_API_KEY and
                                LANGSMITH_API_KEY are not sitting there
                                for any executed command to read

   And one honest admission: mitigations are not isolation. `execute`
   is still a shell on your host; the file confinement applies to the
   agent's file tools, not to what a shell command can reach. The
   challenge in 04 demonstrates that, and 05-extra-sandbox.md is the
   real fix — a rented computer instead of yours.

Run:
    uv run python src/shell_agent.py
    SANDBOX_PROVIDER=daytona uv run python src/shell_agent.py   # extra
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent

load_dotenv()

# There is no USE_FAKE today. The entire point of Day 4 is real
# execution — a fake shell would teach nothing.
PROVIDER = os.getenv("SANDBOX_PROVIDER", "local")

WORK_DIR = Path(__file__).resolve().parent.parent / "work"

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free"),
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = (
    "You are a Python engineer with a shell.\n"
    "Work by manipulating the environment, not by reasoning in your head: "
    "write code to a file with your filesystem tools, run it with execute, "
    "read the actual output, and fix what actually broke. Never claim a "
    "program's output without having executed it.\n"
    "If a command fails, read the error before changing anything, then make "
    "the smallest change that addresses it. If a package is missing, install "
    "it and re-run.\n"
    "Report exactly what the program printed, verbatim, before any commentary."
)


def make_backend():
    """Return (backend, cleanup_fn).

    Two backends, one interface — the calling code in mission.py and
    challenge.py never learns which one it got. The choice of WHERE
    dangerous work happens is configuration (SANDBOX_PROVIDER), not a
    code change, which is exactly where that decision belongs.
    """
    if PROVIDER == "local":
        from deepagents.backends import LocalShellBackend

        WORK_DIR.mkdir(exist_ok=True)

        backend = LocalShellBackend(
            root_dir=str(WORK_DIR),
            virtual_mode=True,
            # inherit_env is False by default; passing PATH explicitly
            # keeps `python` and `pip` findable while leaving every
            # secret in this process's environment out of the shell's.
            env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
        )
        # Nothing to tear down: the "sandbox" is your own computer,
        # which is the uncomfortable part.
        return backend, (lambda: None)

    if PROVIDER == "daytona":
        # EXTRA — see 05-extra-sandbox.md. A rented computer: the
        # boundary moves from a system prompt into infrastructure.
        from daytona import Daytona
        from langchain_daytona import DaytonaSandbox

        sandbox = Daytona().create()
        backend = DaytonaSandbox(sandbox=sandbox)
        # Cleanup MATTERS here — an un-stopped sandbox keeps billing.
        return backend, sandbox.stop

    raise ValueError(f"unknown SANDBOX_PROVIDER: {PROVIDER!r} (use 'local' or 'daytona')")


TASK = (
    "1. Create calculator.py with add, subtract, multiply and divide "
    "functions; divide must raise ValueError on division by zero. "
    "2. Write test_calculator.py with pytest tests, including the "
    "division-by-zero case. "
    "3. Run them with execute (`python -m pytest -q`; pip install pytest "
    "first if it is missing). "
    "4. Fix any failures and re-run until the suite is green. "
    "5. Report the final pytest output exactly as printed."
)


if __name__ == "__main__":
    backend, cleanup = make_backend()
    try:
        agent = create_deep_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            backend=backend,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})
        print(result["messages"][-1].content)
    finally:
        # finally, not "at the end": if the agent raises halfway through,
        # a rented sandbox still has to be stopped.
        cleanup()

    print(f"\nThe files are really on disk — look: ls {WORK_DIR}")
