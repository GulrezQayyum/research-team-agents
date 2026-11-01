import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict
from urllib.parse import urlparse

from config import get_llm, tracker
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from system_b_fixed_pipeline import (
    MAX_REVISIONS,
    critic_node,
    researcher_node,
    writer_node,
)


RECURSION_LIMIT = 20
MAX_SEARCHES_PER_RESEARCHER_PASS = 3
MAX_RUN_TOKENS = 20_000
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0


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
    errors: List[str]
    failed: bool
    failure_reason: str
    tokens_used: int
    token_start: int


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("rate limit", "rate_limit", "429", "timeout", "timed out")
    )


def _retry_delay(attempt: int) -> None:
    time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))


def _valid_url(value: str) -> bool:
    for token in value.split():
        candidate = token.strip("()[]{}<>.,;\"'")
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return True
    return False


def _error_update(state: SupervisorState, message: str) -> Dict[str, Any]:
    print(f"[SUPERVISOR ERROR] {message}")
    return {
        "errors": state["errors"] + [message],
        "failed": True,
        "failure_reason": message,
        "next": "FINISH",
    }


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
        f"revisions_used={state['revisions']}/{MAX_REVISIONS}; "
        f"tokens_used={state['tokens_used']}/{MAX_RUN_TOKENS}; "
        f"errors={len(state['errors'])}; failed={state['failed']}"
    )


def supervisor_node(state: SupervisorState) -> Dict[str, Any]:
    """Propose the next worker, then enforce the pipeline's control invariants."""
    if state["failed"] or state["tokens_used"] >= MAX_RUN_TOKENS:
        reason = state["failure_reason"] or "per-run token budget reached"
        return {
            "next": "FINISH",
            "decision_log": state["decision_log"]
            + [{"proposed": "FINISH", "selected": "FINISH", "reason": reason}],
        }

    try:
        llm = get_llm(temperature=0.0).with_structured_output(
            SupervisorDecision, method="json_mode"
        )
    except Exception as exc:
        return _error_update(state, f"supervisor setup failed: {exc}")
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
    response = None
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content="Return only the requested structured routing decision."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            break
        except Exception as exc:
            if not _is_retryable(exc) or attempt == MAX_RETRY_ATTEMPTS - 1:
                return _error_update(state, f"supervisor failed: {exc}")
            _retry_delay(attempt)
    if response is None:
        return _error_update(state, "supervisor failed without a response")
    _record_usage(response)
    tokens_used = tracker.total_tokens - state.get("token_start", tracker.total_tokens)
    if tokens_used >= MAX_RUN_TOKENS:
        return {
            "next": "FINISH",
            "tokens_used": tokens_used,
            "failure_reason": "per-run token budget reached",
            "failed": True,
            "errors": state["errors"] + ["per-run token budget reached"],
        }
    try:
        proposed = response.next
        reason = response.reason
    except Exception as exc:
        return _error_update(state, f"supervisor returned invalid routing data: {exc}")
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
        "tokens_used": tokens_used,
    }


def route_from_supervisor(state: SupervisorState) -> str:
    return state["next"]


def _worker_update(state: SupervisorState, worker: str) -> Dict[str, Any]:
    if worker == "researcher":
        return researcher_node(state)  # type: ignore[arg-type]
    if worker == "writer":
        return writer_node(state)  # type: ignore[arg-type]
    return critic_node(state)  # type: ignore[arg-type]


def _run_worker_safely(state: SupervisorState, worker: str) -> Dict[str, Any]:
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            update = _worker_update(state, worker)
            if not isinstance(update, dict):
                raise ValueError(f"{worker} returned a non-dict update")
            if worker == "researcher":
                notes = update.get("research_notes")
                if not isinstance(notes, list) or not notes:
                    raise ValueError("researcher returned no research notes")
                if len(notes) > MAX_SEARCHES_PER_RESEARCHER_PASS:
                    raise ValueError("researcher exceeded its search cap")
                if any(not isinstance(note, str) or not _valid_url(note) for note in notes):
                    raise ValueError("researcher returned a finding without a valid URL")
            tokens_used = tracker.total_tokens - state.get(
                "token_start", tracker.total_tokens
            )
            if tokens_used > MAX_RUN_TOKENS:
                return _error_update(state, "per-run token budget exceeded")
            return {**update, "tokens_used": tokens_used}
        except Exception as exc:
            if not _is_retryable(exc) or attempt == MAX_RETRY_ATTEMPTS - 1:
                return _error_update(state, f"{worker} failed: {exc}")
            _retry_delay(attempt)
    return _error_update(state, f"{worker} failed after retries")


def researcher_worker(state: SupervisorState) -> Dict[str, Any]:
    return {
        "route_sequence": state["route_sequence"] + ["researcher"],
        **_run_worker_safely(state, "researcher"),
    }


def writer_worker(state: SupervisorState) -> Dict[str, Any]:
    return {
        "route_sequence": state["route_sequence"] + ["writer"],
        **_run_worker_safely(state, "writer"),
    }


def critic_worker(state: SupervisorState) -> Dict[str, Any]:
    return {
        "route_sequence": state["route_sequence"] + ["critic"],
        **_run_worker_safely(state, "critic"),
    }


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
        "errors": [],
        "failed": False,
        "failure_reason": "",
        "tokens_used": 0,
        "token_start": tracker.total_tokens,
    }
    final_state = dict(initial_state)
    try:
        for update in app.stream(
            initial_state, config={"recursion_limit": RECURSION_LIMIT}
        ):
            for node_update in update.values():
                final_state.update(node_update)
    except GraphRecursionError as exc:
        message = f"recursion limit reached: {exc}"
        final_state.update(
            {
                "failed": True,
                "failure_reason": message,
                "errors": final_state["errors"] + [message],
                "next": "FINISH",
            }
        )
    except Exception as exc:
        message = f"pipeline failed: {exc}"
        final_state.update(
            {
                "failed": True,
                "failure_reason": message,
                "errors": final_state["errors"] + [message],
                "next": "FINISH",
            }
        )
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
