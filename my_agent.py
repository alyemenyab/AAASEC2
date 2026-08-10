import os
import operator
from datetime import datetime
from typing import Annotated, List, Dict
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# Safe fallback for embeddings (works without downloading huge local models)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except ImportError:
    import hashlib
    import random
    from langchain_core.embeddings import Embeddings

    class LocalDeterministicFakeEmbeddings(Embeddings):
        def __init__(self, size: int = 384):
            self.size = size

        def _get_embedding(self, seed: int) -> list[float]:
            rng = random.Random(seed)
            return [rng.gauss(0, 1) for _ in range(self.size)]

        @staticmethod
        def _get_seed(text: str) -> int:
            return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**8

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._get_embedding(seed=self._get_seed(text)) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._get_embedding(seed=self._get_seed(text))

    embeddings = LocalDeterministicFakeEmbeddings(size=384)

# ============================================================
# STEP 0 — GRAPH BUILDING BLOCKS (LangGraph)
# ============================================================
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Check for offline mode flag
USE_FAKE = os.environ.get("USE_FAKE", "0") == "1"

# Detect missing API credentials and fall back to deterministic local mode.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
if not USE_FAKE:
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-or-"):
        print("WARNING: No valid OPENAI_API_KEY found. Falling back to USE_FAKE=1 for offline mode.")
        USE_FAKE = True
    elif not TAVILY_API_KEY or TAVILY_API_KEY.startswith("tvly-"):
        print("WARNING: No valid TAVILY_API_KEY found. Falling back to USE_FAKE=1 for offline mode.")
        USE_FAKE = True


# ============================================================
# STEP 1 — THE STATE (The "Digital Clipboard")
# ============================================================
class AgentState(TypedDict):
    topic: str
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    # REDUCER: operator.add allows nodes to append logs rather than overwrite them
    execution_logs: Annotated[List[str], operator.add]


# ============================================================
# STEP 2 — MODEL, SEARCH TOOL, VECTOR STORE
# ============================================================
if not USE_FAKE:
    # OpenRouter API integration (OpenAI-compatible)
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )
    search_tool = TavilySearch(max_results=3)
else:
    # Deterministic Fakes for offline testing / smoke tests
    llm = None
    search_tool = None

# Local/In-Memory Vector Store for RAG Memory
vector_store = InMemoryVectorStore(embeddings)


# ============================================================
# STEP 3 — STRUCTURED OUTPUT FOR QUALITY SCORE
# ============================================================
class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10, description="Quality score from 1 to 10")
    reasoning: str = Field(description="One-sentence justification for the score")

if llm and not USE_FAKE:
    # Forces the LLM to output structured JSON matching the Pydantic schema
    evaluator = llm.with_structured_output(QualityScore)
else:
    evaluator = None


# ============================================================
# STEP 4 — NODES (Return Partial Dict Updates Only)
# ============================================================
def collect_node(state: AgentState) -> Dict:
    """Search the web. On retries, change the query strategy."""
    iteration = state["iteration_count"] + 1
    
    # Change query per iteration to avoid fetching identical results repeatedly
    if iteration == 1:
        query = f"{state['topic']} architecture enterprise trends"
    elif iteration == 2:
        query = f"{state['topic']} production patterns case studies"
    else:
        query = f"{state['topic']} technical implementations best practices"

    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Step 1 (Iteration {iteration}): Executing search query: '{query}'"
    
    if USE_FAKE or not search_tool:
        results = [
            {"title": f"Fake Source {iteration}.1", "content": f"Enterprise agents require state machines and memory mechanisms for {state['topic']}."},
            {"title": f"Fake Source {iteration}.2", "content": f"Flow engineering outperforms simple prompting in complex agentic workflows."}
        ]
    else:
        try:
            raw_res = search_tool.invoke({"query": query})
            results = raw_res.get("results", []) if isinstance(raw_res, dict) else []
        except Exception as e:
            results = [{"title": "Error Search", "content": f"Search failed: {str(e)}"}]

    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": [log_msg]
    }


def store_memory_node(state: AgentState) -> Dict:
    """Store retrieved content into the vector datastore for RAG retrieval."""
    docs = []
    for item in state.get("collected_data", []):
        content = item.get("content", "")
        title = item.get("title", "Unknown")
        if content:
            docs.append(Document(page_content=content, metadata={"source": title}))
    
    if docs:
        vector_store.add_documents(docs)
        
    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Step 2: Stored {len(docs)} documents into in-memory vector store."
    return {"execution_logs": [log_msg]}


def analyze_node(state: AgentState) -> Dict:
    """Analyze collected sources and perform RAG similarity queries against memory."""
    analyzed = []

    for item in state.get("collected_data", []):
        content = item.get("content", "")

        # RAG Memory Retrieval: retrieve context from past runs/sources
        rag_context = "No previous context."
        if not USE_FAKE:
            try:
                similar_memories = vector_store.similarity_search(content, k=1)
                rag_context = similar_memories[0].page_content if similar_memories else rag_context
            except ImportError:
                rag_context = "No previous context."
            except Exception:
                rag_context = "No previous context."

        if USE_FAKE or not llm:
            analysis = f"Key Insight: {content[:120]}... [Context Match: {rag_context[:60]}...]"
        else:
            prompt = (
                f"Analyze the following research snippet regarding '{state['topic']}'.\n"
                f"Snippet: {content}\n"
                f"Related Memory Context: {rag_context}\n"
                f"Provide a 2-sentence key enterprise insight."
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            analysis = response.content

        analyzed.append({"title": item.get("title", "Source"), "summary": analysis})

    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Step 3: Analyzed {len(analyzed)} sources with RAG memory enrichment."
    return {"analyzed_data": analyzed, "execution_logs": [log_msg]}


def evaluate_node(state: AgentState) -> Dict:
    """Evaluate research quality using structured output."""
    iteration = state["iteration_count"]
    
    if USE_FAKE or not evaluator:
        # Simulate improvement over iterations for testing
        score = 8 if iteration >= 2 else 5
        reasoning = "Information reaches acceptable depth." if score >= 7 else "Needs deeper technical detail."
    else:
        summaries = "\n".join([a["summary"] for a in state.get("analyzed_data", [])])
        prompt = (
            f"Topic: {state['topic']}\n"
            f"Collected Summaries:\n{summaries}\n\n"
            f"Rate the depth, accuracy, and technical quality of this research from 1 to 10."
        )
        try:
            result = evaluator.invoke([HumanMessage(content=prompt)])
            score = result.score
            reasoning = result.reasoning
        except Exception:
            score = 7  # Fallback
            reasoning = "Default evaluation score due to model output formatting fallback."

    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Step 4: Quality Evaluation Score = {score}/10. Reasoning: {reasoning}"
    return {"quality_score": score, "execution_logs": [log_msg]}


def report_node(state: AgentState) -> Dict:
    """Generate final comprehensive report."""
    analyzed_data = state.get("analyzed_data", [])
    
    if USE_FAKE or not llm:
        report = (
            f"# Executive Report: {state['topic']}\n\n"
            f"**Quality Score:** {state['quality_score']}/10 after {state['iteration_count']} iterations.\n\n"
            f"## Key Findings:\n" + "\n".join([f"- **{a['title']}**: {a['summary']}" for a in analyzed_data])
        )
    else:
        findings = "\n".join([f"- {a['title']}: {a['summary']}" for a in analyzed_data])
        prompt = (
            f"Synthesize a executive summary report on '{state['topic']}' using these key findings:\n{findings}\n"
            f"Final score: {state['quality_score']}/10."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        report = response.content

    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Step 5: Compiled final executive report."
    return {"final_report": report, "execution_logs": [log_msg]}


def audit_node(state: AgentState) -> Dict:
    """Audit completion stats."""
    log_msg = (
        f"[{datetime.now().strftime('%H:%M:%S')}] Step 6: Workflow Completed Successfully. "
        f"Total Iterations: {state['iteration_count']}, Final Quality: {state['quality_score']}/10."
    )
    return {"execution_logs": [log_msg]}


# ============================================================
# STEP 5 — CONDITIONAL EDGE (Loop Control)
# ============================================================
def quality_router(state: AgentState) -> str:
    """
    Route flow based on quality score and iteration cap.
    - Quality >= 7 -> proceed to report.
    - Iterations >= 3 -> forced exit to report (prevents infinite loop).
    - Otherwise -> return to collect.
    """
    score = state.get("quality_score", 0)
    iterations = state.get("iteration_count", 0)
    
    if score >= 7 or iterations >= 3:
        return "report"
    return "collect"


# ============================================================
# STEP 6 — WIRE THE GRAPH
# ============================================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report", report_node)
workflow.add_node("audit", audit_node)

# Entry Point
workflow.add_edge(START, "collect")

# Linear Pipeline
workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")

# Conditional Loop Edge
workflow.add_conditional_edges(
    "evaluate",
    quality_router,
    {
        "collect": "collect",
        "report": "report"
    }
)

# Terminal Edges
workflow.add_edge("report", "audit")
workflow.add_edge("audit", END)


# ============================================================
# STEP 7 — COMPILE, VISUALIZE, & STREAM EXECUTION
# ============================================================
if __name__ == "__main__":
    # Compile with memory persistence
    app = workflow.compile(checkpointer=InMemorySaver())

    # 1. Print Mermaid Diagram (Paste into https://mermaid.live)
    print("\n" + "="*50)
    print("MERMAID GRAPH DIAGRAM:")
    print("="*50)
    print(app.get_graph().draw_mermaid())
    print("="*50 + "\n")

    # Initial State Definition
    initial_state = {
        "topic": "Enterprise Agentic AI Systems",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }

    config = {"configurable": {"thread_id": "run-1"}}

    print("STARTING WORKFLOW EXECUTION STREAM...\n")
    
    # Stream execution node-by-node
    for chunk in app.stream(initial_state, config, stream_mode="values"):
        if "execution_logs" in chunk and chunk["execution_logs"]:
            print(chunk["execution_logs"][-1])

    # Fetch final state snapshot
    final_state = app.get_state(config).values

    print("\n" + "="*50)
    print("FINAL EXECUTIVE REPORT")
    print("="*50)
    print(final_state.get("final_report", "No report generated."))
    print("="*50)
