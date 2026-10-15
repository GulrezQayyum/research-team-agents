import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import json
from pathlib import Path
import hashlib

load_dotenv()

def run_llm_check():
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0
    )

    return llm

print("--- STEP 1: Basic Call & Token Tracking ---")

messages = [
    SystemMessage(content="You are a concise, helpful technical assistant."),
    HumanMessage(content="Explain the difference between a process and a thread in two sentences.")
]

response = run_llm_check().invoke(messages)

print(f"Content:\n{response.content}\n")
print(f"Usage Metadata: {response.usage_metadata}")

total_input_tokens = response.usage_metadata.get("input_tokens", 0)
total_output_tokens = response.usage_metadata.get("output_tokens", 0)

print("\n --- Step 2: Defined Web Search Tool with Disk Cache ---")

CACHE_DIR = Path(".tool_cache")
CACHE_DIR.mkdir(exist_ok=True)

@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo for real-time information, recent news, or factual facts.
    
    Args:
        query: Clear, specific search terms to look up.
    """
    query_hash = hashlib.md5(query.lower().strip().encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{query_hash}.json"

    if cache_file.exists():
        print(f"[CACHE HIT] Returning cached DuckDuckGo result for: '{query}'")
        return cache_file.read_text(encoding="utf-8")

    print(f"[CACHE MISS] Fetching real DuckDuckGo search for: '{query}'")
    
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        
    if not results:
        formatted_result = "No search results found."
    else:
        formatted_result = "\n\n".join(
            [f"Title: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}" for r in results]
        )
    
    cache_file.write_text(formatted_result, encoding="utf-8")
    return formatted_result

print(f"Tool Name: {web_search.name}")
print(f"Tool Description (what LLM sees): {web_search.description}")

print("--- Summary ---")
print(f"Cumulative Input Tokens : {total_input_tokens}")
print(f"Cumulative Output Tokens: {total_output_tokens}")
print(f"Total Tokens Used       : {total_input_tokens + total_output_tokens}")