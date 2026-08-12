"""
DAY 3 — Agent implementation.

READ FIRST:  ../01-deep-agents.md

The one line that matters in this file is not a line of code, it is the
FUNCTION BOUNDARY:

    build_agent() -> object with .ainvoke({"messages": [...]})

api.py calls exactly that and nothing else. Behind it today sits a Deep
Agent (or a deterministic fake); tomorrow it could be Day 2's supervisor
graph or a hand-wired StateGraph, and neither the API, the Dockerfile,
nor any future frontend would notice. Everything downstream depends on
the boundary, never on the implementation.

Run:
    USE_FAKE=1 uv run python src/agent.py          # no keys, instant
    uv run python src/agent.py                     # real agent + tools
    uv run python src/agent.py "your own question" # ad-hoc smoke test
"""

import ast
import operator
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

USE_FAKE = os.getenv("USE_FAKE", "0") == "1"

# day3/ — the root the agent's filesystem tools are confined to.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free")


# ────────────────────────────────────────────────────────────────
# TWO DELIBERATELY BORING TOOLS
# ────────────────────────────────────────────────────────────────
# Boring is the point: Day 3 is about everything AROUND the agent —
# the boundary, the port, the container, the protocol. A clever tool
# here would only distract from that.

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Walk an arithmetic AST by hand.

    Why not eval()? Because eval() would make this tool a shell: names,
    attribute access, and calls all become reachable, and `__import__`
    is three characters away from arbitrary code execution. An allowlist
    of node types is the difference between "a calculator" and "a remote
    code execution endpoint I shipped by accident". Giving an agent real
    execution is a decision with a whole day attached to it — Day 4.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression such as '2 * (3 + 4) ** 2'.

    Supports + - * / // % ** and parentheses over numbers only. No
    variables, no function calls, no attribute access.
    """
    return _eval_node(ast.parse(expression, mode="eval").body)


def current_time() -> str:
    """Return the current date and time in UTC, ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SYSTEM_PROMPT = (
    "You are the AAASEC2 course agent: a careful research and analysis "
    "assistant.\n"
    "Tool discipline: you are bad at arithmetic and you do not know the "
    "current time, so use the calculate and current_time tools for those "
    "instead of answering from memory — even when the sum looks easy.\n"
    "Skills: before answering a request that names a document type (a "
    "brief, a commit message, a report), check your skills and follow the "
    "matching one exactly, including its structure and its limits.\n"
    "Say plainly when you do not know something. Do not invent sources."
)


# ────────────────────────────────────────────────────────────────
# THE FAKE
# ────────────────────────────────────────────────────────────────
# USE_FAKE=1 keeps the ENTIRE day runnable with no API key: FastAPI,
# Docker, compose, MCP, the A2A round trip. Only the model is faked —
# every piece of plumbing being taught today is the real thing.

class FakeAgent:
    """Deterministic stand-in with the real agent's interface."""

    class _Message:
        """Mimics a LangChain message: api.py only reads .content."""

        def __init__(self, content: str) -> None:
            self.content = content

    @staticmethod
    def _text_of(message) -> str:
        return message["content"] if isinstance(message, dict) else message.content

    async def ainvoke(self, payload, config=None):
        prompt = self._text_of(payload["messages"][-1])
        skills = sorted(p.parent.name for p in (PROJECT_ROOT / "skills").glob("*/SKILL.md"))
        reply = (
            f"[FAKE AGENT] received: {prompt[:160]!r}\n"
            f"With a real key I would plan, call my tools, and consult my "
            f"skills ({', '.join(skills) or 'none found'}).\n"
            f"Proof the tools themselves are real: calculate('17*23') = "
            f"{calculate('17*23')}, current_time() = {current_time()}."
        )
        return {"messages": list(payload["messages"]) + [self._Message(reply)]}


# ────────────────────────────────────────────────────────────────
# THE BOUNDARY
# ────────────────────────────────────────────────────────────────

def build_agent():
    """Return something with .ainvoke({"messages": [...]}) -> {"messages": [...]}."""
    if USE_FAKE:
        return FakeAgent()

    # Imported lazily so that USE_FAKE=1 needs neither the model client
    # nor a key — the fake path must stay cheap to start.
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

    # FilesystemBackend: the agent's ls/read_file/write_file/edit_file
    # tools act on real disk under root_dir, and virtual_mode makes
    # every path the agent sees relative to that root — so "/skills/"
    # means day3/skills/ and "/etc/passwd" means nothing at all.
    #
    # Note what this backend does NOT include: an execute tool. File
    # access is not code execution, and code execution needs a boundary
    # to happen inside. That boundary is tomorrow's subject.
    backend = FilesystemBackend(root_dir=str(PROJECT_ROOT), virtual_mode=True)

    return create_deep_agent(
        model=llm,
        tools=[calculate, current_time],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        skills=["/skills/"],  # virtual path, resolved under root_dir
    )


if __name__ == "__main__":
    import asyncio

    question = " ".join(sys.argv[1:]) or "What is 17 * 23, and what time is it right now?"

    agent = build_agent()
    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    )

    print(f"Q: {question}\n")
    print(result["messages"][-1].content)
