# ============================================================
# DAY 2 LAB — Build a Multi-Agent Research Team
# ============================================================
# Completed lab. Every TODO from the skeleton is implemented below;
# the teaching comments are kept so the file still reads as a lesson.
#
# WHAT CHANGES FROM DAY 1 — read this table twice:
#
#   Day 1 (single agent)              Day 2 (multi-agent)
#   ─────────────────────             ─────────────────────────────
#   nodes = Python functions          nodes = LLM agents w/ personas
#   routing = your if/else            routing = supervisor LLM decides
#   one prompt for everything         one system prompt PER agent
#   tools available everywhere        tools SCOPED (only researcher
#                                       can search the web)
#   loop = quality-score retry        loop = critic sends draft back
#                                       to writer for revision
#
# What does NOT change: State + Nodes + Edges. A multi-agent system
# is STILL just a StateGraph.
#
# The system built here (the SUPERVISOR pattern):
#
#              ┌──────────── supervisor ─────────────┐
#              │       (LLM decides who's next)      │
#     ┌────────┼───────────┬───────────┬─────────────┤
#     ↓        ↓           ↓           ↓             ↓
#  researcher  analyst    writer     critic       FINISH
#     │        │           │           │             ↓
#     └────────┴───────────┴───────────┘            END
#          (every worker reports back to the supervisor)
#
# Two run modes:
#   USE_FAKE=1  → no API keys, deterministic fakes. The fake critic
#                 rejects the first draft so the revision loop is
#                 visible: writer → critic → writer → critic → FINISH.
#   default     → real OpenRouter LLM + Tavily search (.env keys).
# ============================================================

import os
import operator
from datetime import datetime
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

# STEP 0 — same building blocks as Day 1
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

USE_FAKE = os.getenv("USE_FAKE", "0") == "1"

MAX_REVISIONS = 2      # cap on writer↔critic loops
MAX_TURNS = 12         # cap on total supervisor decisions


# ============================================================
# STEP 1 — SHARED STATE: the team's "blackboard"
# ============================================================
# Day 1's state was a data PIPELINE (each field filled once, in
# order). Day 2's state is a BLACKBOARD: every agent reads all of
# it and writes only its own section; the supervisor reads it to
# decide who goes next.
#
# WHY research_notes APPENDS BUT draft OVERWRITES:
# research_notes is evidence — a second research pass should add to
# the record, not erase it, so it gets operator.add. draft is the
# single current answer; if it appended, every revision would stack
# a new draft onto the old one, the critic would review a growing
# pile of contradictory text, and "the draft" would stop having a
# meaning. Reducers encode intent, not just plumbing.

class TeamState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]   # append-only evidence
    analysis: str                                        # analyst overwrites
    draft: str                                           # writer overwrites
    critique: str                                        # critic overwrites
    revision_count: int
    turn_count: int
    next_agent: str                                      # supervisor's decision
    execution_logs: Annotated[List[str], operator.add]   # append-only trace


# ============================================================
# STEP 2 — STRUCTURED ROUTING DECISION
# ============================================================
# Day 1: structured output produced a quality SCORE.
# Day 2: structured output produces a ROUTING DECISION — this is
# the trick that turns an LLM into a supervisor. Literal[...] means
# the model CANNOT invent an agent that doesn't exist: an invalid
# route becomes a validation error instead of a KeyError deep in
# the graph.

class RouterDecision(BaseModel):
    """The supervisor's choice of who acts next."""
    next_agent: Literal["researcher", "analyst", "writer", "critic", "FINISH"]
    reason: str = Field(description="One sentence explaining the choice")


# ============================================================
# STEP 3 — ONE LLM, FOUR PERSONAS (+ tools scoped per agent)
# ============================================================
# A team doesn't need four models — it needs four SYSTEM PROMPTS.
# Each persona says what the agent DOES and what it MUST NOT do:
# the boundaries between agents live in the prompts, so write them
# sharp. Overlapping personas is how a "team" silently degrades
# into four copies of the same generalist.
#
# TOOL SCOPING: only researcher_node touches search_tool. If the
# critic could search, it would review the draft against fresh
# facts the writer never saw — the loop would never converge, and
# a "revision" would really be a whole new task.

SUPERVISOR_PROMPT = (
    "You are the SUPERVISOR of a four-person research team: researcher, "
    "analyst, writer, critic. You do no work yourself — you only choose who "
    "acts next, based on the status report you are given.\n"
    "Normal order: researcher → analyst → writer → critic.\n"
    "If the critique starts with REVISE and the revision budget is not spent, "
    "send the writer back in. If the critique is APPROVED, or the revision "
    "budget is spent, answer FINISH. Never pick an agent whose input is not "
    "ready yet (no analyst before research, no critic before a draft)."
)

PERSONAS = {
    "researcher": (
        "You are the team's RESEARCHER. Your only job is gathering raw facts "
        "from the search results you are handed. Produce 3-5 short factual "
        "bullets, each ending with its source URL in parentheses. "
        "MUST NOT: interpret, rank, recommend, or write prose — the analyst "
        "and the writer own those. If the sources disagree, report both."
    ),
    "analyst": (
        "You are the team's ANALYST. You are handed research notes and you "
        "find the shape in them: the trend, the tension, the implication. "
        "Produce 3-4 tight sentences of analysis. "
        "MUST NOT: search for new facts, invent numbers that are not in the "
        "notes, or draft the final report."
    ),
    "writer": (
        "You are the team's WRITER. Turn the analysis into an executive brief "
        "of about 150 words: a headline, 2-3 findings, one recommendation. "
        "Cite the sources carried in the research notes. If a critique is "
        "present, treat it as a work order and fix every point in it. "
        "MUST NOT: add claims that appear nowhere in the notes or analysis."
    ),
    "critic": (
        "You are the team's CRITIC. Judge the draft against the research "
        "notes only — accuracy, specificity, completeness, sourcing. "
        "Reply with the single word APPROVED if it holds up. Otherwise reply "
        "'REVISE:' followed by numbered, concrete fixes the writer can act on. "
        "MUST NOT: rewrite the draft yourself or ask for facts nobody gathered."
    ),
}


if USE_FAKE:
    # ---------- deterministic fakes: the whole team, offline ----------
    # The fakes stand in for the two things that cost money: the
    # worker LLM and the supervisor LLM. Everything else — state,
    # reducers, guardrails, graph wiring — is the real thing.

    _FAKE_SEARCH = [
        {"title": "Supervisor pattern in production agent teams",
         "content": "Teams with an explicit router outperform unsupervised swarms.",
         "url": "https://example.com/supervisor"},
        {"title": "Tool scoping and agent error rates",
         "content": "Restricting tools per role cut wrong-tool calls by roughly half.",
         "url": "https://example.com/scoping"},
        {"title": "Review loops and output quality",
         "content": "A single critic pass caught most factual slips before delivery.",
         "url": "https://example.com/critique"},
    ]

    class FakeTeam:
        """Canned per-persona output; the critic rejects exactly once."""

        def __init__(self) -> None:
            self.critic_calls = 0

        def respond(self, role: str, _prompt: str) -> str:
            if role == "researcher":
                return "\n".join(
                    f"- {r['title']}: {r['content']} ({r['url']})" for r in _FAKE_SEARCH
                )
            if role == "analyst":
                return (
                    "The sources agree on one thing: coordination, not model size, is "
                    "what makes an agent team useful. Explicit routing and per-role tool "
                    "scoping are the two levers that show up in every result. The cost "
                    "side is under-reported — more agents means more calls and more "
                    "places to fail, so the coordination has to buy something real."
                )
            if role == "writer":
                return (
                    "HEADLINE: Supervised agent teams beat solo agents — when the task "
                    "needs specialists.\n"
                    "Findings: (1) an explicit supervisor outperforms unsupervised swarms "
                    "(example.com/supervisor); (2) scoping tools per role roughly halved "
                    "wrong-tool calls (example.com/scoping); (3) one critic pass caught "
                    "most factual slips (example.com/critique).\n"
                    "Recommendation: pilot a supervisor-pattern team on one workflow that "
                    "genuinely has distinct roles, and measure calls and latency against "
                    "today's single agent before going wider."
                )
            if role == "critic":
                self.critic_calls += 1
                if self.critic_calls == 1:
                    return ("REVISE: 1) attach the source URL to every finding; "
                            "2) quantify finding (2) instead of saying 'fewer errors'; "
                            "3) say what the recommendation would cost.")
                return "APPROVED"
            return ""

    _fake_team = FakeTeam()

    def run_persona(role: str, user_content: str) -> str:
        return _fake_team.respond(role, user_content)

    def supervisor_decide(state: TeamState) -> RouterDecision:
        """Rule-based stand-in that follows the same policy as the prompt."""
        if not state["research_notes"]:
            return RouterDecision(next_agent="researcher", reason="Blackboard is empty.")
        if not state["analysis"]:
            return RouterDecision(next_agent="analyst", reason="Notes are in, nothing analyzed.")
        if not state["draft"]:
            return RouterDecision(next_agent="writer", reason="Analysis is in, no draft yet.")
        if not state["critique"]:
            return RouterDecision(next_agent="critic", reason="A fresh draft needs review.")
        if state["critique"].startswith("REVISE") and state["revision_count"] < MAX_REVISIONS:
            return RouterDecision(next_agent="writer", reason="Critic asked for fixes.")
        return RouterDecision(next_agent="FINISH", reason="Draft approved or budget spent.")

else:
    # ---------- real providers ----------
    from langchain_openai import ChatOpenAI
    from langchain_tavily import TavilySearch

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

    # SCOPED: this handle is used in exactly one node (researcher_node).
    search_tool = TavilySearch(max_results=4)

    supervisor_llm = llm.with_structured_output(RouterDecision)

    def run_persona(role: str, user_content: str) -> str:
        response = llm.invoke(
            [SystemMessage(content=PERSONAS[role]), HumanMessage(content=user_content)]
        )
        return response.content

    def supervisor_decide(state: TeamState) -> RouterDecision:
        return supervisor_llm.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT),
             HumanMessage(content=build_status_summary(state))]
        )


# ============================================================
# STEP 4 — THE SUPERVISOR NODE (the piece Day 1 didn't have)
# ============================================================

def build_status_summary(state: TeamState) -> str:
    """STATUS, not CONTENT.

    The supervisor is answering one question — who goes next — and that
    question is answered by which sections exist, not by what they say.
    Pasting the full draft and notes in here would cost thousands of
    tokens per turn and bury the one line that actually drives the
    decision (the critique verdict). The critique is the single
    exception: its first word IS the routing signal.
    """
    notes = state["research_notes"]
    critique = state["critique"]
    verdict = critique.split(":", 1)[0].strip()[:40] if critique else "none"

    return (
        f"Task: {state['task']}\n"
        f"Research notes: {len(notes)} block(s)\n"
        f"Analysis: {'present' if state['analysis'] else 'missing'}\n"
        f"Draft: {'present' if state['draft'] else 'missing'}\n"
        f"Latest critique verdict: {verdict}\n"
        f"Revisions used: {state['revision_count']} of {MAX_REVISIONS}\n"
        f"Supervisor turns used: {state['turn_count']} of {MAX_TURNS}\n"
        "Who acts next?"
    )


def supervisor_node(state: TeamState):
    """The LLM proposes; this function disposes."""
    turn = state["turn_count"] + 1

    # GUARDRAIL (a) — absolute turn cap. Checked BEFORE the LLM call:
    # once we are over budget there is nothing left to ask about.
    if turn > MAX_TURNS:
        decision = RouterDecision(next_agent="FINISH", reason=f"Turn cap {MAX_TURNS} reached.")
    else:
        decision = supervisor_decide(state)

        # GUARDRAIL (b) — the revision budget. A model that likes the
        # sound of "one more pass" can loop writer↔critic forever, so
        # code, not the prompt, ends the loop. This is Day 1's
        # iteration_count wearing a supervisor's hat.
        if (
            decision.next_agent in ("writer", "critic")
            and state["revision_count"] >= MAX_REVISIONS
            and state["draft"]
        ):
            decision = RouterDecision(
                next_agent="FINISH",
                reason=f"Revision cap {MAX_REVISIONS} reached; shipping the current draft.",
            )

    return {
        "next_agent": decision.next_agent,
        "turn_count": turn,
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] supervisor (turn {turn}): "
            f"→ {decision.next_agent} — {decision.reason}"
        ],
    }


# ============================================================
# STEP 5 — WORKER AGENT NODES
# ============================================================
# Each worker: read the blackboard → act in persona → return a
# PARTIAL update containing ONLY its own section.

def researcher_node(state: TeamState):
    """Search the web (ONLY this agent may) and condense to notes."""
    if USE_FAKE:
        notes = run_persona("researcher", f"Task: {state['task']}")
    else:
        results = search_tool.invoke({"query": state["task"]})["results"]
        raw = "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:300]} ({r.get('url', '')})"
            for r in results
        )
        notes = run_persona("researcher", f"Task: {state['task']}\n\nSearch results:\n{raw}")

    return {
        # a LIST, because research_notes is append-only
        "research_notes": [notes],
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] researcher: {len(notes)} chars of notes gathered"
        ],
    }


def analyst_node(state: TeamState):
    """Turn raw notes into analysis. Never searches."""
    analysis = run_persona(
        "analyst",
        f"Task: {state['task']}\n\nResearch notes:\n" + "\n\n".join(state["research_notes"]),
    )
    return {
        "analysis": analysis,
        "execution_logs": [f"[{datetime.now():%H:%M:%S}] analyst: analysis written"],
    }


def writer_node(state: TeamState):
    """Write the draft — or revise it when a critique is waiting."""
    revising = state["critique"].startswith("REVISE")

    prompt = (
        f"Task: {state['task']}\n\n"
        f"Analysis:\n{state['analysis']}\n\n"
        f"Research notes:\n" + "\n\n".join(state["research_notes"])
    )
    if revising:
        prompt += (
            f"\n\nYour previous draft:\n{state['draft']}\n\n"
            f"Critique you must address point by point:\n{state['critique']}"
        )

    draft = run_persona("writer", prompt)

    return {
        "draft": draft,
        # WHY RESET critique: it describes the draft that just got
        # replaced. Leave it and the supervisor sees "REVISE" on the
        # very next turn, sends the writer back to fix problems that
        # are already fixed, and burns the revision budget in a loop
        # driven by a stale verdict. Clearing it is how the blackboard
        # stays truthful about the CURRENT draft.
        "critique": "",
        "revision_count": state["revision_count"] + (1 if revising else 0),
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] writer: "
            + (f"revision {state['revision_count'] + 1}" if revising else "first draft")
        ],
    }


def critic_node(state: TeamState):
    """Review the draft against the research notes."""
    critique = run_persona(
        "critic",
        f"Task: {state['task']}\n\nResearch notes:\n"
        + "\n\n".join(state["research_notes"])
        + f"\n\nDraft to review:\n{state['draft']}",
    )
    critique = critique.strip()

    # Normalize: downstream code (and guardrail b) branch on the first
    # word, so an LLM that answers "Approved." must not break routing.
    verdict = "APPROVED" if critique.upper().startswith("APPROVED") else "REVISE"
    if verdict == "REVISE" and not critique.upper().startswith("REVISE"):
        critique = f"REVISE: {critique}"

    return {
        "critique": critique,
        "execution_logs": [f"[{datetime.now():%H:%M:%S}] critic: {verdict}"],
    }


# ============================================================
# STEP 6 — ROUTING FUNCTION + WIRE THE GRAPH
# ============================================================
# The conditional-edge function is now TRIVIAL — the decision was
# already made, inside a node, by the supervisor. Compare Day 1,
# where all the logic lived in quality_router: the intelligence
# MOVED from the edge into a node. That is the whole pattern.

def route_from_supervisor(state: TeamState) -> str:
    return state["next_agent"]


WORKERS = ["researcher", "analyst", "writer", "critic"]

workflow = StateGraph(TeamState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    },
)

# Every worker reports BACK to the supervisor — hub and spoke.
for worker in WORKERS:
    workflow.add_edge(worker, "supervisor")


# ============================================================
# STEP 7 — COMPILE, VISUALIZE, RUN
# ============================================================

if __name__ == "__main__":
    app = workflow.compile(checkpointer=InMemorySaver())

    print("=" * 62)
    print("GRAPH (paste into https://mermaid.live) — a STAR, not a chain:")
    print("=" * 62)
    print(app.get_graph().draw_mermaid())

    initial_state = {
        "task": "Should our company adopt multi-agent AI systems in 2026?",
        "research_notes": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "turn_count": 0,
        "next_agent": "",
        "execution_logs": [],
    }

    config = {"configurable": {"thread_id": "team-run-1"}}

    print("\n" + "=" * 62)
    print(f"RUN (USE_FAKE={USE_FAKE}, MAX_REVISIONS={MAX_REVISIONS}, MAX_TURNS={MAX_TURNS})")
    print("=" * 62)

    final_state = None
    for chunk in app.stream(initial_state, config, stream_mode="values"):
        final_state = chunk
        if chunk["execution_logs"]:
            print(chunk["execution_logs"][-1])

    print("\n" + "=" * 62)
    print("FINAL DRAFT")
    print("=" * 62)
    print(final_state["draft"])

    print("\n" + "=" * 62)
    print(
        f"STATS: turns={final_state['turn_count']} "
        f"revisions={final_state['revision_count']} "
        f"verdict={final_state['critique'][:40] or '(cleared)'}"
    )
    print("=" * 62)


# ============================================================
# SELF-CHECK — answered
# ============================================================
# [x] Supervisor pattern in one sentence: workers are specialists that
#     never talk to each other, and one LLM node decides after every
#     step which specialist runs next until it says FINISH.
# [x] route_from_supervisor only READS state — the decision happened
#     in supervisor_node.
# [x] research_notes appends (evidence accumulates), draft overwrites
#     (there is only ever one current answer).
# [x] The writer resets critique; leaving it stale would re-trigger a
#     revision of a draft that no longer has the problem.
# [x] Only researcher_node references search_tool.
# [x] Both guardrails present: turn cap before the LLM call, revision
#     cap after it. EXPERIMENT 2: remove (a) and force the critic to
#     always answer REVISE — (b) still stops it; remove (b) as well and
#     LangGraph's recursion limit raises GraphRecursionError.
# [x] The Mermaid output is a star with supervisor at the center.
# [x] Day 1's single agent is the better design for a fixed pipeline
#     with a machine-checkable bar — e.g. "summarize this document to
#     under 200 words": one call, one check, one retry. Four agents
#     would add four LLM calls, latency, and three new failure modes
#     to buy nothing. Coordination has to earn its cost.
#
# EXPERIMENT 1 (MAX_REVISIONS = 0): the critic still runs once, but
# guardrail (b) fires immediately after and the first draft ships
# uncritiqued — the review loop becomes decorative.
# EXPERIMENT 3 (vague analyst): the damage does not stay local. The
# writer only sees the analysis, so a vague analysis produces a vague
# draft; the critic compares that draft to the notes, asks for
# specifics the writer cannot supply, and the team burns its whole
# revision budget without improving. Bad personas fail downstream.
# ============================================================
