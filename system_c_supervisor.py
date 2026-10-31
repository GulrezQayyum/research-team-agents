import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict

from config import get_llm, tracker
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from system_b_fixed_pipeline import (
    MAX_REVISIONS,
    critic_node,
    researcher_node,
    writer_node,
)


WorkerName = Literal["researcher", "writer", "critic", "FINISH"]


class SupervisorDecision(BaseModel):
    next: WorkerName = Field(description="The next worker to run, or FINISH.")
    reason: str = Field(description="A short reason for this routing decision.")


class SupervisorState(TypedDict):
    topic: str
    research_notes: List[str]
    draft: str
    critique: str
    approved: bool
    revisions: int
    next: WorkerName
    decision_log: List[Dict[str, str]]
    route_sequence: List[str]


def _record_usage(message: Any) -> None:
    usage = getattr(message, "usage_metadata", None)
    if usage:
        tracker.add_usage(usage)


def _compact_status(state: SupervisorState) -> str:
    last_verdict = (
        "approved"
        if state["approved"]
        else "not approved"
        if state["critique"]
        else "not reviewed"
    )
    return (
        f"topic={state['topic']}; notes={len(state['research_notes'])}; "
        f"draft_exists={bool(state['draft'])}; last_critique={last_verdict}; "
        f"revisions_used={state['revisions']}/{MAX_REVISIONS}"
    )


def supervisor_node(state: SupervisorState) -> Dict[str, Any]:
    """Propose the next worker, then enforce the pipeline's control invariants."""
    llm = get_llm(temperature=0.0).with_structured_output(
        SupervisorDecision, method="json_mode"
    )
    prompt = f"""You supervise a research pipeline.

Worker descriptions:
- researcher: gather concise factual notes about the topic; choose it when no research notes exist.
- writer: create or revise the article from the notes; choose it when notes exist but no draft exists, or when critique requests revision.
- critic: evaluate the draft for accuracy, specificity, and clarity; choose it only when a draft exists and it needs review.
- FINISH: stop only when the draft is complete and approved, or the revision cap has been reached.

Current compact status:
{_compact_status(state)}

Choose exactly one next worker and give a short reason.
Return only this JSON shape:
{{"next": "researcher" | "writer" | "critic" | "FINISH", "reason": "short reason"}}"""
    response = llm.invoke(
        [
            SystemMessage(content="Return only the requested structured routing decision."),
            HumanMessage(content=prompt),
        ]
    )
    _record_usage(response)
    proposed = response.next
    reason = response.reason
    enforced = proposed
    override_reason = ""

    if state["revisions"] >= MAX_REVISIONS:
        enforced = "FINISH"
        override_reason = "revision cap reached"
    elif proposed == "FINISH" and not state["draft"]:
        enforced = "writer"
        override_reason = "cannot finish without a draft"
    elif proposed == "critic" and not state["draft"]:
        enforced = "writer"
        override_reason = "cannot route to critic without a draft"
    elif proposed == "researcher" and state["research_notes"]:
        enforced = "writer" if not state["draft"] else "critic"
        override_reason = "research is already complete"
    elif proposed == "FINISH" and not state["approved"]:
        enforced = "critic"
        override_reason = "draft must be approved before finishing"

    decision_reason = reason
    if override_reason:
        decision_reason = f"{reason} Override: {override_reason}."

    decision = {
        "proposed": proposed,
        "selected": enforced,
        "reason": decision_reason,
    }
    print(
        f"[SUPERVISOR] proposed={proposed} selected={enforced} "
        f"reason={decision_reason}"
    )
    return {
        "next": enforced,
        "decision_log": state["decision_log"] + [decision],
    }


def route_from_supervisor(state: SupervisorState) -> str:
    return state["next"]


def _worker_update(state: SupervisorState, worker: str) -> Dict[str, Any]:
    if worker == "researcher":
        return researcher_node(state)  # type: ignore[arg-type]
    if worker == "writer":
        return writer_node(state)  # type: ignore[arg-type]
    return critic_node(state)  # type: ignore[arg-type]


def researcher_worker(state: SupervisorState) -> Dict[str, Any]:
    return {"route_sequence": state["route_sequence"] + ["researcher"], **_worker_update(state, "researcher")}


def writer_worker(state: SupervisorState) -> Dict[str, Any]:
    return {"route_sequence": state["route_sequence"] + ["writer"], **_worker_update(state, "writer")}


def critic_worker(state: SupervisorState) -> Dict[str, Any]:
    return {"route_sequence": state["route_sequence"] + ["critic"], **_worker_update(state, "critic")}


builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_worker)
builder.add_node("writer", writer_worker)
builder.add_node("critic", critic_worker)
builder.add_edge(START, "supervisor")
for worker in ("researcher", "writer", "critic"):
    builder.add_edge(worker, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    },
)
app = builder.compile()


def run_supervisor_pipeline(topic: str) -> SupervisorState:
    initial_state: SupervisorState = {
        "topic": topic,
        "research_notes": [],
        "draft": "",
        "critique": "",
        "approved": False,
        "revisions": 0,
        "next": "researcher",
        "decision_log": [],
        "route_sequence": [],
    }
    final_state = dict(initial_state)
    for update in app.stream(initial_state):
        for node_update in update.values():
            final_state.update(node_update)
    return final_state


@dataclass
class RouteComparison:
    topic: str
    supervisor_routes: List[str]
    system_b_routes: List[str]
    same_path: bool


def compare_with_system_b(topics: List[str]) -> List[RouteComparison]:
    comparisons: List[RouteComparison] = []
    for topic in topics:
        state = run_supervisor_pipeline(topic)
        b_routes = ["researcher", "writer", "critic"]
        b_routes += ["writer", "critic"] * max(0, state["revisions"] - 1)
        comparisons.append(
            RouteComparison(
                topic=topic,
                supervisor_routes=state["route_sequence"],
                system_b_routes=b_routes,
                same_path=state["route_sequence"] == b_routes,
            )
        )
    return comparisons


if __name__ == "__main__":
    topics = [
        "quantum computing",
        "vertical farming",
        "robotics",
        "renewable energy",
        "large language models",
    ]
    started = time.perf_counter()
    for comparison in compare_with_system_b(topics):
        print(
            f"{comparison.topic}: supervisor={comparison.supervisor_routes}; "
            f"system_b={comparison.system_b_routes}; "
            f"same_path={comparison.same_path}"
        )
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}")
