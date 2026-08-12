"""
DAY 3 — HTTP API.

READ FIRST:  ../03-fastapi-openresponses.md
             ../09-a2a.md   (for the agent card endpoint)

Until now, "using the agent" meant importing this package. From here on
it means speaking HTTP to a port — which changes who can use it from
"me" to "anything with a network stack", including other students'
agents this afternoon.

The contract is an OpenResponses SUBSET: plain-text `input` in, one
assistant message out. A deliberate subset, because the lesson is the
boundary (API contract  ≠  agent implementation), not spec coverage.

Run:
    USE_FAKE=1 uv run uvicorn src.api:app --port 8000 --reload

    curl http://localhost:8000/healthz
    curl -X POST http://localhost:8000/v1/responses \
         -H 'Content-Type: application/json' -d '{"input": "hi"}'
    curl http://localhost:8000/.well-known/agent-card.json
    open http://localhost:8000/docs
"""

import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Works both as a package (uvicorn src.api:app) and as a loose script.
try:
    from .agent import MODEL_NAME, build_agent
except ImportError:  # pragma: no cover
    from agent import MODEL_NAME, build_agent

load_dotenv()

STUDENT_NAME = os.getenv("STUDENT_NAME", "anonymous")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000").rstrip("/")

app = FastAPI(
    title=f"{STUDENT_NAME} agent",
    description="AAASEC2 Day 3 — an agent exposed as a network service.",
    version="0.1.0",
)

# BUILT ONCE, AT IMPORT — not per request.
# Per-request construction would re-create the model client, re-read the
# skills directory, and re-open the filesystem backend on every single
# call: latency on every response, for nothing. The agent holds no
# per-user state between requests, so one instance serves everybody.
agent = build_agent()


# ────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SHAPES
# ────────────────────────────────────────────────────────────────
# Declaring these as Pydantic models is what makes /docs exist and what
# turns a malformed request into a clean 422 instead of a KeyError in
# the middle of the handler. That is what a *contract* buys you.

class ResponsesRequest(BaseModel):
    input: str = Field(description="The user's prompt, as plain text.")
    model: str | None = Field(default=None, description="Advisory only in this subset.")


# ────────────────────────────────────────────────────────────────
# ENDPOINTS
# ────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    """Liveness probe.

    Trivial today; on Docker, compose, and the shared server this is the
    endpoint everything else asks before trusting the service.
    """
    return {"status": "ok"}


@app.post("/v1/responses")
async def create_response(request: ResponsesRequest):
    """OpenResponses subset: plain-text in, one assistant message out."""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request.input}]}
    )

    last = result["messages"][-1]
    text = last["content"] if isinstance(last, dict) else last.content

    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": request.model or MODEL_NAME,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
def agent_card():
    """A2A discovery document.

    STUDENT_NAME and PUBLIC_URL come from the environment, so the SAME
    image produces a DIFFERENT card per student on the shared server —
    configuration at run time, never baked in at build time.

    `url` is the only address a peer needs; they learn where to send
    work FROM the card instead of hardcoding it. That indirection is
    the protocol.
    """
    return {
        "protocolVersion": "1.0",
        "name": f"{STUDENT_NAME}-agent",
        "description": (
            "Research and analysis agent. Turns a topic into a structured "
            "one-page brief with sourced findings and an explicit confidence "
            "line, and writes Conventional Commits messages from a described "
            "change. Exact arithmetic and current UTC time come from tools, "
            "not from the model's memory."
        ),
        "url": f"{PUBLIC_URL}/v1/responses",
        "version": "0.1.0",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "research-brief",
                "name": "Research brief",
                "description": (
                    "One-page executive brief on a technical topic: headline, "
                    "context, exactly three findings, a recommendation, and a "
                    "stated confidence level. Under 200 words."
                ),
                "tags": ["research", "writing", "summary", "brief"],
            },
            {
                "id": "commit-message",
                "name": "Commit message",
                "description": (
                    "Conventional Commits message from a description of a "
                    "change: typed, scoped, imperative subject under 50 "
                    "characters, body only when the reason is not obvious."
                ),
                "tags": ["git", "writing", "engineering"],
            },
        ],
    }
