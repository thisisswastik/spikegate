"""
agent/graph.py — LangGraph graph assembling all five nodes.

Graph topology:
  context_fetch → policy_reasoner → bounds_gate → explainer → audit_writer

All edges are unconditional — the graph always runs all five nodes.
Failure handling is done inside each node (graceful degradation), not via
conditional edges, so the audit trail is always written.

Usage
-----
from agent.graph import build_graph
from data_gen.schema import DetectorOutput

graph = build_graph()
result = graph.invoke({"detector_output": detector_output_obj, "errors": []})
print(result["final_action"])
print(result["explanation"])
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.context_fetch import context_fetch
from agent.nodes.policy_reasoner import policy_reasoner
from agent.nodes.bounds_gate import bounds_gate
from agent.nodes.explainer import explainer
from agent.nodes.audit_writer import audit_writer


def build_graph() -> StateGraph:
    """Build and compile the SpikeGate LangGraph agent graph."""

    # Initialise with our state schema
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("context_fetch", context_fetch)
    graph.add_node("policy_reasoner", policy_reasoner)
    graph.add_node("bounds_gate", bounds_gate)
    graph.add_node("explainer", explainer)
    graph.add_node("audit_writer", audit_writer)

    # Linear chain: each node feeds the next
    graph.add_edge("context_fetch", "policy_reasoner")
    graph.add_edge("policy_reasoner", "bounds_gate")
    graph.add_edge("bounds_gate", "explainer")
    graph.add_edge("explainer", "audit_writer")
    graph.add_edge("audit_writer", END)

    # Entry point
    graph.set_entry_point("context_fetch")

    return graph.compile()


# Module-level singleton (lazy-initialised on first import)
_graph = None


def get_graph():
    """Return the compiled graph singleton, building it if needed."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
