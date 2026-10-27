# Research Team Agents

An implementation of a multi-turn ReAct (Reasoning + Acting) loop and structured output supervisor router built with **LangChain**, **LangGroq**, and **Pydantic**.

##  Project Overview

This repository demonstrates key LLM agent concepts:
1. **Basic Model Invocations & Token Tracking:** Tracking cumulative usage across standard completions.
2. **Resilient Tool Caching:** A disk-backed cache (`.tool_cache/`) wrapping DuckDuckGo (`ddgs`) search that prevents redundant API requests and ignores failures/rate limits.
3. **Manual ReAct Loop:** Multi-turn tool invocation loop with loop bounds, unknown tool handling, and fallback execution.
4. **Structured Routing Benchmark:** Router classification utilizing `with_structured_output(method="json_mode", include_raw=True)` to measure schema parsing success and routing accuracy across standard worker nodes (`search_agent`, `code_agent`, `final_responder`).

---

##  Repository Structure

```text
research-team-agents/
├── .tool_cache/       # Local disk cache for search tool results (ignored by git)
├── config.py          # Centralized LLM setup & TokenTracker class
├── tools.py           # Custom web search tool definition with exception handling & cache
├── benchmark_groq.py  # Benchmark script
├── main.py            # Main entry point running Steps 1 through 4
├── requirements.txt   # Python project dependencies
└── .gitignore         # Ignores .venv, .env, and .tool_cache/