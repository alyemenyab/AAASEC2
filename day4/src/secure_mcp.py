"""
DAY 4 — Authenticated MCP server.

READ FIRST:  ../02-mcp-auth.md

Yesterday:  MCP URL            -> access
Today:      MCP URL + identity -> access

Two words that must not blur:

    authentication = who are you?           (the bearer token)
    authorization  = what may you access?   (the scopes on that token)

StaticTokenVerifier is a DEV tool — predefined tokens, no infrastructure,
never do this in production, where you would verify real JWTs from an
identity provider. The architecture is identical either way, and the
architecture is the lesson.

Run:
    uv run python src/secure_mcp.py       # port 8002
Verify:
    uv run python src/check_auth.py       # the auth matrix, in another terminal
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth import require_scopes
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

load_dotenv()

# Two identities, two scope sets. The student token is not a "weaker
# password" — it is a different principal that a different set of tools
# is willing to answer.
verifier = StaticTokenVerifier(
    tokens={
        os.getenv("MCP_STUDENT_TOKEN", "student-secret-token"): {
            "client_id": "student",
            "scopes": ["read:public"],
        },
        os.getenv("MCP_ADMIN_TOKEN", "admin-secret-token"): {
            "client_id": "admin",
            "scopes": ["read:public", "read:internal"],
        },
    }
)

mcp = FastMCP("alyemenyab Secure Tools", auth=verifier)


# ────────────────────────────────────────────────────────────────
# PUBLIC — any valid token
# ────────────────────────────────────────────────────────────────

@mcp.tool
def get_server_time() -> str:
    """Return the server's current UTC time in ISO-8601 format.

    Public: authentication still required (a valid token), authorization
    trivial (no scope beyond being someone).
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ────────────────────────────────────────────────────────────────
# PROTECTED — requires the read:internal scope
# ────────────────────────────────────────────────────────────────
# Watch what happens in check_auth.py when the student token calls one
# of these: not "403 Forbidden" but "Unknown tool". FastMCP filters
# unauthorized tools out of DISCOVERY entirely — authorization shapes
# what you can see exists, not just what you can call. A caller cannot
# probe for the shape of data they were never allowed to know about.

@mcp.tool(auth=require_scopes("read:internal"))
def get_internal_report() -> dict:
    """Quarterly revenue and cost figures. Internal use only."""
    return {
        "quarter": "Q2-2026",
        "currency": "SAR",
        "months": [
            {"month": "April", "revenue": 412_000, "costs": 287_500},
            {"month": "May", "revenue": 468_300, "costs": 301_200},
            {"month": "June", "revenue": 501_750, "costs": 356_900},
        ],
        "headcount": 34,
        "classification": "confidential",
    }


@mcp.tool(auth=require_scopes("read:internal"))
def get_lab_inventory() -> dict:
    """Drone-lab parts inventory: stock levels, unit costs, reorder points.

    Internal use only. Quantities are current as of the last stock count.
    """
    return {
        "lab": "AAASEC2 aerial robotics lab",
        "currency": "SAR",
        "counted_on": "2026-08-10",
        "items": [
            {"sku": "BAT-6S-5200", "name": "6S LiPo 5200mAh", "quantity": 14,
             "unit_cost": 385.0, "reorder_level": 10},
            {"sku": "MTR-2207-1750", "name": "2207 brushless motor", "quantity": 6,
             "unit_cost": 142.5, "reorder_level": 12},
            {"sku": "PRP-5045-CF", "name": "5045 carbon propeller", "quantity": 48,
             "unit_cost": 27.0, "reorder_level": 20},
            {"sku": "FCU-H7-PRO", "name": "H7 flight controller", "quantity": 3,
             "unit_cost": 690.0, "reorder_level": 4},
            {"sku": "LDR-TOF-40", "name": "40m ToF rangefinder", "quantity": 9,
             "unit_cost": 455.0, "reorder_level": 6},
            {"sku": "TET-CBL-30M", "name": "30m power tether", "quantity": 2,
             "unit_cost": 1_240.0, "reorder_level": 3},
        ],
        "classification": "confidential",
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8002)
