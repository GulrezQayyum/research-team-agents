import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict

from config import get_llm, tracker
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from tools import web_search


MAX_TOOL_STEPS = 3
MAX_REVISIONS = 2


class PipelineState(TypedDict):
    topic: str
    research_notes: List[str]
    draft: str
    critique: str
    approved: bool
    revisions: int


def _record_usage(message: Any) -> None:
    usage = getattr(message, "usage_metadata", None)
    if usage:
        tracker.add_usage(usage)


def researcher_node(state: PipelineState) -> Dict[str, Any]:
    """Gather concise research notes with a bounded tool-calling loop."""
    llm = get_llm(temperature=0.0).bind_tools([web_search])
    messages = [
        SystemMessage(
            content=(
                "You are a succinct researcher. Gather key facts about the topic "
                "using the search tool when needed. Keep the final notes factual, "
                "short, and free of unsupported claims."
            )
        ),
        HumanMessage(content=f"Research the topic: {state['topic']}"),
    ]
    notes: List[str] = []

    for _ in range(MAX_TOOL_STEPS):
        response = llm.invoke(messages)
        _record_usage(response)
        messages.append(response)

        if not isinstance(response, AIMessage) or not response.tool_calls:
            if response.content:
                notes.append(str(response.content))
            break

        for tool_call in response.tool_calls:
            if tool_call["name"] != web_search.name:
                raise ValueError(f"Unsupported research tool: {tool_call['name']}")
            tool_output = web_search.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
            notes.append(f"Research result: {tool_output}")

    if not notes:
        raise RuntimeError("Researcher produced no notes.")
    return {"research_notes": notes}


def writer_node(state: PipelineState) -> Dict[str, Any]:
    """Write or revise a two-paragraph article grounded in the research notes."""
    llm = get_llm(temperature=0.7)
    research_summary = "\n".join(f"- {note}" for note in state["research_notes"])
    prompt = (
        "Topic: "
        f"{state['topic']}\n"
        "Research notes:\n"
        f"{research_summary}\n"
        "Write a clear, factual two-paragraph article based strictly on these notes."
    )
    if state["critique"]:
        prompt += (
            "\n\nPrevious critique:\n"
            f"{state['critique']}\n"
            "Explicitly address every issue in the critique and remove unsupported "
            "or vague claims."
        )

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert technical writer. Do not introduce facts "
                    "that are absent from the research notes."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    _record_usage(response)
    if not response.content:
        raise RuntimeError("Writer produced an empty draft.")
    return {"draft": str(response.content)}


def _parse_critic_response(content: str) -> Dict[str, Any]:
    clean_json = content.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean_json)
    if not isinstance(result, dict) or not isinstance(result.get("approved"), bool):
        raise ValueError("Critic response must contain a boolean 'approved' field.")
    feedback = result.get("feedback", "")
    if not isinstance(feedback, str):
        raise ValueError("Critic response must contain string 'feedback'.")
    return {"approved": result["approved"], "feedback": feedback}


def critic_node(state: PipelineState) -> Dict[str, Any]:
    """Evaluate the draft against accuracy, specificity, and clarity criteria."""
    llm = get_llm(temperature=0.0)
    research_summary = "\n".join(f"- {note}" for note in state["research_notes"])
    prompt = f"""Target topic: {state['topic']}
Research ground truth:
{research_summary}

Draft to review:
{state['draft']}

Rubric:
1. Accuracy: every major claim must align with the research notes.
2. Specificity: claims must be precise rather than vague generalizations.
3. Clarity: the tone must be objective and cohesive.

Return valid JSON with exactly two keys:
{{"approved": true or false, "feedback": "specific fixes, or empty when approved"}}
Approve only when all rubric standards are met."""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a strict editorial evaluator. Return only valid JSON."
            ),
            HumanMessage(content=prompt),
        ]
    )
    _record_usage(response)
    try:
        result = _parse_critic_response(str(response.content))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Critic returned invalid JSON: {exc}") from exc

    return {
        "approved": result["approved"],
        "critique": result["feedback"],
        "revisions": state["revisions"] + 1,
    }


def should_continue(state: PipelineState) -> Literal["writer", "__end__"]:
    """Stop when approved or when the revision budget has been exhausted."""
    if state["approved"] or state["revisions"] >= MAX_REVISIONS:
        return END
    return "writer"


builder = StateGraph(PipelineState)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges("critic", should_continue, {"writer": "writer", END: END})
app = builder.compile()


@dataclass
class ExecutionMetrics:
    topic: str
    total_time_sec: float
    total_revisions: int
    final_approved: bool
    notes_count: int
    final_draft: str


def run_and_stream_pipeline(topic: str, run_wrapper: Any = None) -> PipelineState:
    """Run the graph and print each state update as it is produced."""
    initial_state: PipelineState = {
        "topic": topic,
        "research_notes": [],
        "draft": "",
        "critique": "",
        "approved": False,
        "revisions": 0,
    }
    final_state = dict(initial_state)
    print(f"\n--- Starting System B pipeline for: {topic!r} ---")

    for update in app.stream(initial_state):
        for node_name, node_update in update.items():
            final_state.update(node_update)
            print(f"[STREAM UPDATE] {node_name}: {list(node_update)}")

    if run_wrapper is not None:
        run_wrapper(final_state)
    return final_state


class Stage3Wrapper:
    """Run the pipeline for multiple topics and collect timing metrics."""

    @staticmethod
    def execute_and_log(
        topics: List[str], pipeline_runner=run_and_stream_pipeline
    ) -> List[ExecutionMetrics]:
        metrics_log: List[ExecutionMetrics] = []
        for topic in topics:
            start = time.perf_counter()
            final_state = pipeline_runner(topic)
            metrics_log.append(
                ExecutionMetrics(
                    topic=topic,
                    total_time_sec=round(time.perf_counter() - start, 3),
                    total_revisions=final_state["revisions"],
                    final_approved=final_state["approved"],
                    notes_count=len(final_state["research_notes"]),
                    final_draft=final_state["draft"],
                )
            )
        return metrics_log


if __name__ == "__main__":
    for metric in Stage3Wrapper.execute_and_log(["quantum computing"]):
        print(
            f"\nFinal approval: {metric.final_approved}; "
            f"revisions: {metric.total_revisions}\n{metric.final_draft}"
        )
