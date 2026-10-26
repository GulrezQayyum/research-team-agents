# system_a_baseline.py
from datetime import datetime
import json
import operator
import os
from pathlib import Path
import re
import time
from typing import Annotated, Any, Dict, List, TypedDict
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from config import get_llm
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools import web_search

MAX_SEARCHES = 4
RESULTS_DIR = Path(".results")
RESULTS_DIR.mkdir(exist_ok=True)

CURRENT_DATE_STR = datetime.now().strftime("%B %d, %Y")


# =====================================================================
# State & Prompts
# =====================================================================
class AgentState(TypedDict):
  messages: Annotated[List[BaseMessage], add_messages]
  retries: Annotated[int, operator.add]


SYSTEM_PROMPT = SystemMessage(
    content=(
        f"Today is {CURRENT_DATE_STR}.\n"
        f"You are a research assistant. Research the topic using AT MOST"
        f" {MAX_SEARCHES} search tool calls.\n"
        "Rules:\n"
        "1. Write a 400 to 600 word detailed report based strictly on retrieved"
        " research.\n"
        "2. Provide explicit URL citations (e.g., [Title](URL)) for every claim"
        " made.\n"
        "3. NEVER cite a source or URL that did not explicitly appear in actual"
        " search results.\n"
        "4. State research limitations clearly if search results fail or are"
        " incomplete. Do NOT add uncited claims."
    )
)

# LLM Instances
llm_default = get_llm(temperature=0.0)
llm_temp_retry = get_llm(temperature=0.3)

tools = [web_search]
tool_node = ToolNode(tools)


# =====================================================================
# Graph Nodes (Task 1.3: Corrective Retry inside llm_node)
# =====================================================================
def llm_node(state: AgentState) -> Dict[str, Any]:
  messages = state["messages"]
  if not isinstance(messages[0], SystemMessage):
    messages = [SYSTEM_PROMPT] + messages

  attempt = 0
  max_retries = 2
  local_messages = list(messages)
  retries_count = 0

  while attempt <= max_retries:
    try:
      current_llm = llm_default if attempt < 2 else llm_temp_retry
      llm_bound = current_llm.bind_tools(tools)
      response = llm_bound.invoke(local_messages)

      return {"messages": [response], "retries": retries_count}
    except Exception as e:
      err_str = str(e).lower()
      if any(
          k in err_str
          for k in [
              "tool_use_failed",
              "tool call validation failed",
              "failed to parse",
          ]
      ):
        attempt += 1
        retries_count = 1
        if attempt <= max_retries:
          retry_instruction = HumanMessage(
              content=(
                  "Error: Tool call validation failed. Call web_search with"
                  " exactly one argument named 'query' as a plain string."
              )
          )
          local_messages.append(retry_instruction)
        else:
          raise e
      else:
        raise e


def final_report_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
  synthesis_instruction = HumanMessage(
      content=(
          f"Current Date: {CURRENT_DATE_STR}. You have reached your maximum"
          " search budget. Synthesize a 400-600 word report using ONLY the"
          " search results retrieved above. Cite explicit URLs from tool"
          " outputs and state any search limitations clearly."
      )
  )
  prompt_messages = state["messages"] + [synthesis_instruction]
  response = llm_default.invoke(prompt_messages)
  return {"messages": [response]}


def route_tools(state: AgentState) -> str:
  messages = state["messages"]
  last_message = messages[-1]

  tool_messages_executed = sum(
      1 for msg in messages if isinstance(msg, ToolMessage)
  )

  if tool_messages_executed >= MAX_SEARCHES:
    return "final_report"

  if isinstance(last_message, AIMessage) and last_message.tool_calls:
    return "tools"

  return END


# Build Workflow
workflow = StateGraph(AgentState)
workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)
workflow.add_node("final_report", final_report_node)

workflow.set_entry_point("llm")
workflow.add_conditional_edges(
    "llm", route_tools, {"tools": "tools", "final_report": "final_report", END: END}
)
workflow.add_edge("tools", "llm")
workflow.add_edge("final_report", END)

graph_app = workflow.compile()


# =====================================================================
# Task 3: URL Normalization & Citation Honesty
# =====================================================================
def normalize_url(url: str) -> str:
  url = url.strip().rstrip(".,;!?:')\"]")
  parsed = urlparse(url)
  netloc = parsed.netloc.lower()
  path = parsed.path.rstrip("/")

  query_params = parse_qs(parsed.query)
  filtered_params = {
      k: v
      for k, v in query_params.items()
      if not k.startswith("utm_") and k != "ref"
  }
  new_query = urlencode(filtered_params, doseq=True)

  return urlunparse(("", netloc, path, "", new_query, ""))


def extract_urls_from_tool_messages(messages: List[BaseMessage]) -> List[str]:
  urls = []
  url_pattern = re.compile(r'https?://[^\s><"\'\]\)]+')
  for msg in messages:
    if isinstance(msg, ToolMessage):
      found_urls = url_pattern.findall(str(msg.content))
      urls.extend(found_urls)
  return sorted(list(set(urls)))


def evaluate_citation_honesty(
    report_text: str, retrieved_urls: List[str]
) -> Dict[str, Any]:
  raw_urls = re.findall(r'https?://[^\s><"\'\]\)]+', report_text)
  cited_urls = [r.rstrip(".,;!?:')\"]") for r in raw_urls]
  unique_cited = sorted(list(set(cited_urls)))

  norm_retrieved = {normalize_url(u) for u in retrieved_urls}

  n_cited = len(unique_cited)
  n_valid = sum(1 for u in unique_cited if normalize_url(u) in norm_retrieved)

  no_citations = n_cited == 0
  ratio = (n_valid / n_cited) if n_cited > 0 else None

  paragraphs = [
      p.strip()
      for p in report_text.split("\n\n")
      if p.strip() and not p.startswith("#")
  ]
  uncited_paragraphs = sum(
      1 for p in paragraphs if not re.search(r"https?://", p)
  )

  return {
      "n_cited": n_cited,
      "n_valid": n_valid,
      "ratio": round(ratio, 2) if ratio is not None else "N/A",
      "no_citations": no_citations,
      "uncited_paragraphs": uncited_paragraphs,
  }


# =====================================================================
# Task 1.2 & 1.5: Stream Execution & Ordered Classification
# =====================================================================
def run_agent_with_metrics(topic: str, run_id: int) -> Dict[str, Any]:
  start_time = time.time()

  initial_user_msg = HumanMessage(
      content=(
          f"Research topic: {topic}\nUse at most {MAX_SEARCHES} searches, write"
          " a 400-600 word report, and include full URL citations from tool"
          " outputs."
      )
  )

  initial_state = {
      "messages": [SYSTEM_PROMPT, initial_user_msg],
      "retries": 0,
  }

  error_msg = None
  failure_reason = None
  last_seen_state = {}

  try:
    for event in graph_app.stream(
        initial_state, stream_mode="values", config={"recursion_limit": 20}
    ):
      last_seen_state = event
    final_messages = last_seen_state.get("messages", [])
  except Exception as e:
    error_msg = str(e)
    # Task 1.2 Order of Classification
    if isinstance(e, GraphRecursionError):
      failure_reason = "recursion"
    elif any(
        k in error_msg.lower()
        for k in [
            "tool_use_failed",
            "tool call validation failed",
            "failed to parse",
        ]
    ):
      failure_reason = "tool_call_invalid"
    elif "429" in error_msg or "rate limit" in error_msg.lower():
      failure_reason = "rate_limit"
    elif "timed out" in error_msg.lower():
      failure_reason = "timeout"
    else:
      failure_reason = "graph_exception"

    final_messages = last_seen_state.get("messages", [])

  wall_time = round(time.time() - start_time, 2)

  llm_calls = 0
  tool_calls_count = 0
  input_tokens = 0
  output_tokens = 0
  final_report = ""

  for msg in final_messages:
    if isinstance(msg, AIMessage):
      llm_calls += 1
      if msg.tool_calls:
        tool_calls_count += len(msg.tool_calls)

      usage = getattr(msg, "usage_metadata", None) or {}
      if usage:
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

      if not msg.tool_calls and msg.content:
        final_report = msg.content

  retrieved_urls = extract_urls_from_tool_messages(final_messages)
  citation_eval = evaluate_citation_honesty(final_report, retrieved_urls)

  status_failed = False
  if failure_reason:
    status_failed = True
    print(
        f"[FAILED RUN ERROR PREVIEW] Reason: {failure_reason} | Exception:"
        f" {str(error_msg)[:300]}"
    )
  elif not final_report or len(final_report.strip()) < 50:
    status_failed = True
    failure_reason = "empty_report"

  run_payload = {
      "run_id": run_id,
      "topic": topic,
      "timestamp": datetime.now().isoformat(),
      "status_failed": status_failed,
      "failure_reason": failure_reason,
      "wall_time_sec": wall_time,
      "retries": last_seen_state.get("retries", 0),
      "llm_calls": llm_calls,
      "tool_calls": tool_calls_count,
      "input_tokens": input_tokens,
      "output_tokens": output_tokens,
      "retrieved_urls": retrieved_urls,
      "citation_eval": citation_eval,
      "error_message": error_msg,
      "report": final_report,
  }

  safe_topic = re.sub(r"[^a-zA-Z0-9]", "_", topic[:25])
  out_file = (
      RESULTS_DIR / f"run_{safe_topic}_run{run_id}_{int(time.time())}.json"
  )
  out_file.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

  return run_payload


if __name__ == "__main__":
  test_topics = [
      "Recent breakthroughs in solid-state battery commercialization",
      "Developments in AI hardware inference accelerators",
  ]

  print(f"=== SYSTEM A BENCHMARK EXECUTION ({CURRENT_DATE_STR}) ===\n")
  for topic in test_topics:
    print(f"\n--- Topic: '{topic}' ---")
    for run_idx in range(1, 4):
      res = run_agent_with_metrics(topic, run_idx)
      print(
          f"Run {run_idx} Complete | Time: {res['wall_time_sec']}s | Failed:"
          f" {res['status_failed']} ({res['failure_reason'] or 'None'}) |"
          f" Retries: {res['retries']} | Citation Ratio:"
          f" {res['citation_eval']['ratio']}"
      )
      time.sleep(2)