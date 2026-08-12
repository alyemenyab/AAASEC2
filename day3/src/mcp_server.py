"""
DAY 3 — MCP server.

READ FIRST:  ../06-fastmcp.md   then   ../07-skills-over-mcp.md

Until now the tools in agent.py lived inside one Python process, usable
by exactly one agent. This file puts them on the network behind a
discovery protocol, so any MCP-speaking client — Claude Desktop, Cursor,
another student's agent, a script — can list and call them.

Keep the two categories apart:

    TOOLS  = actions another agent can CALL   (@mcp.tool)
    SKILLS = knowledge another agent can READ (SkillsDirectoryProvider)

The provider does NOT execute skills remotely. It transports the
knowledge; the receiving agent's runtime decides whether to activate it;
some execution environment on that side runs anything dangerous. Three
responsibilities, three places — half of Day 4 is already in that
sentence.

Run:
    uv run python src/mcp_server.py        # serves on :8001

Verify with a client (not curl — MCP is a protocol, not REST):
    async with Client("http://localhost:8001/mcp") as c:
        await c.list_tools()
        await c.list_resources()
"""

import re
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider

# The docstrings and type hints below ARE the wire interface: they are
# what a remote agent reads to decide whether to call the tool and with
# what. Write them like documentation, because that is what they become.
try:
    from .agent import calculate as _calculate
except ImportError:  # pragma: no cover — running as a loose script
    from agent import calculate as _calculate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

mcp = FastMCP("alyemenyab Tools")


@mcp.tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the number.

    Accepts + - * / // % ** and parentheses over numeric literals, e.g.
    '2 * (3 + 4) ** 2' -> 98.0. Rejects anything else, including
    variables and function calls — this is a calculator, not a shell.
    """
    # Same AST-walking implementation the local agent uses; sharing it
    # keeps one definition of "safe arithmetic" in the repo instead of
    # two that drift apart.
    return _calculate(expression)


@mcp.tool
def word_stats(text: str) -> dict:
    """Count words, sentences, and unique words in a block of text.

    Returns words, sentences, unique_words, characters, and
    average_word_length (rounded to two decimals). Useful for checking a
    draft against a length limit before returning it — for example the
    200-word cap in the research-brief skill.
    """
    words = re.findall(r"[A-Za-z']+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    total_letters = sum(len(w) for w in words)

    return {
        "words": len(words),
        "sentences": len(sentences),
        "unique_words": len({w.lower() for w in words}),
        "characters": len(text),
        "average_word_length": round(total_letters / len(words), 2) if words else 0.0,
    }


# ONE LINE turns a folder of Markdown into network-discoverable
# resources: skill://research-brief/SKILL.md, skill://commit-message/…,
# each with a _manifest carrying file sizes and hashes. Yesterday these
# were files on a laptop; now they are versioned artifacts another agent
# can find and download.
mcp.add_provider(SkillsDirectoryProvider(roots=PROJECT_ROOT / "skills"))


if __name__ == "__main__":
    # 0.0.0.0, not 127.0.0.1: inside a container, binding to loopback
    # means nothing outside the container can ever reach the port.
    mcp.run(transport="http", host="0.0.0.0", port=8001)
