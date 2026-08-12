"""
DAY 3 — A2A discovery + delegation client.

READ FIRST:  ../09-a2a.md
USED IN:     ../10-challenge.md

MCP is an agent reaching DOWN into capabilities. A2A is independent
agents reaching ACROSS to each other — different owners, different
stacks, no shared code. All this client is allowed to know about a peer
is one base URL; everything else it learns from the Agent Card.

Usage:
    uv run python src/a2a_client.py http://localhost:8000 "write a research brief on tethered drones"
    uv run python src/a2a_client.py http://<server>:<their-port> "<task their card advertises>"
"""

import sys

import httpx

CARD_PATH = "/.well-known/agent-card.json"
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def discover(peer_base_url: str) -> dict:
    """Fetch and print a peer's Agent Card, then return it."""
    base = peer_base_url.rstrip("/")
    response = httpx.get(base + CARD_PATH, timeout=TIMEOUT)
    response.raise_for_status()
    card = response.json()

    print(f"discovered: {card.get('name', '<unnamed>')} "
          f"(A2A {card.get('protocolVersion', '?')}, v{card.get('version', '?')})")
    print(f"  endpoint: {card.get('url')}")
    print(f"  {card.get('description', '').strip()}")

    skills = card.get("skills", [])
    print(f"  skills ({len(skills)}):")
    for skill in skills:
        tags = ", ".join(skill.get("tags", []))
        print(f"    - {skill.get('id')}: {skill.get('description', '')[:90]}"
              + (f"  [{tags}]" if tags else ""))
    if not skills:
        print("    (none advertised — anything you send is a guess)")

    return card


def delegate(card: dict, task: str) -> str:
    """Send a task to the endpoint the CARD names, and return the reply.

    Note what is absent: any string like "/v1/responses". The endpoint is
    read out of card["url"]. If the peer moves their agent to another
    host, port, or path, they publish a new card and this client keeps
    working — that indirection IS the protocol. Hardcode the path and
    you have reinvented a private integration with extra steps.
    """
    url = card.get("url")
    if not url:
        raise ValueError("agent card has no 'url' field — nothing to delegate to")

    response = httpx.post(url, json={"input": task}, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    # Walk the OpenResponses envelope down to the text, tolerating a
    # peer whose implementation is a little different from ours.
    for item in payload.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                return part["text"]

    raise ValueError(f"no output_text in peer reply: {str(payload)[:200]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__.strip().split("Usage:")[-1].strip())
        sys.exit(1)

    peer_url, task = sys.argv[1], " ".join(sys.argv[2:])

    print("=" * 62)
    card = discover(peer_url)

    print("=" * 62)
    print(f"delegating: {task}\n")
    print(delegate(card, task))
    print("=" * 62)
