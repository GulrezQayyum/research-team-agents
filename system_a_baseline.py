import time
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_llm, tracker
from tools import web_search

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(
    content="You are a research assistant. Research the topic using AT MOST 4 search tool calls.\n"
    "Rules:\n"
    "1. Write a 400 to 600 word detailed report based on your research.\n"
    "2. Provide explicit URL citations (e.g., [Title](URL)) for every claim.\n"
    "3. NEVER cite a source or URL that did not appear in actual search results.\n"
    "4. If search results return empty or error, synthesize whatever partial knowledge you have or state limitations clearly."
)

llm = get_llm()
tools = [web_search]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

def llm_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Calls the model bound with tools, injecting system prompt on first turn."""
    messages = state["messages"]
    if not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def route_tools(state: AgentState) -> str:
    """Routing function: go to 'tools' if tool calls exist, else END."""
    messages = state["messages"]
    last_message = messages[-1]

    ai_tool_turns = sum(
        1 for msg in messages if isinstance(msg, AIMessage) and msg.tool_calls
    )

    if ai_tool_turns >= 4:
        return END
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)

workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("llm")

workflow.add_conditional_edges(
    "llm",
    route_tools,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "llm")

graph_app = workflow.compile()

def run_agent_with_metrics(topic: str, max_searches: int = 4) -> Dict[str, Any]:
    """Runs System A baseline and captures runtime metrics and token totals."""
    start_time = time.time()
    
    initial_user_msg = HumanMessage(
        content=f"Research topic: {topic}. Write a 400-600 word report with full URL citations." 
    )
    
    error_msg = None
    final_state = None
    
    try:
        final_state = graph_app.invoke({"messages": [initial_user_msg]},
                                       config={"recursion_limit": 15}
                                       )
    except Exception as e:
        error_msg = str(e)

    wall_time = round(time.time() - start_time, 2)
    
    llm_calls = 0
    tool_calls_count = 0
    input_tokens = 0
    output_tokens = 0
    final_report = ""

    if final_state and "messages" in final_state:
        for msg in final_state["messages"]:
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

    return {
        "topic": topic,
        "wall_time_sec": wall_time,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error_msg,
        "report": final_report
    }

if __name__ == "__main__":
    test_topics = [
        "Recent breakthroughs in solid-state battery commercialization",
        "Developments in AI hardware inference accelerators"
    ]

    print("=== SYSTEM A BASELINE BENCHMARK ===\n")
    for topic in test_topics:
        res = run_agent_with_metrics(topic)
        print("=" * 60)
        print(f"TOPIC       : {res['topic']}")
        print(f"WALL TIME   : {res['wall_time_sec']}s")
        print(f"LLM CALLS   : {res['llm_calls']} | TOOL CALLS: {res['tool_calls']}")
        print(f"TOKENS      : {res['input_tokens']} in / {res['output_tokens']} out")
        print(f"ERROR       : {res['error']}")
        print("-" * 60)
        print("REPORT PREVIEW:")
        print(res['report'][:400] + "...\n")
