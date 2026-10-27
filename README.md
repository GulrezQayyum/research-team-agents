# Research Team Agents

An enterprise-grade research agent framework built with **LangGraph**, **LangChain**, **LangGroq**, and **Pydantic**. This repository demonstrates declarative state graph architectures, multi-turn ReAct loops, structured output routing, resilient tool caching, and rigorous citation evaluation.

---

##  Features

- **Declarative Graph Architecture (System A Baseline):** Built using LangGraph `StateGraph` with a two-node loop (`llm_node`, `tool_node`), custom conditional edge routing, and a dedicated budget-exhausted report synthesis node.
- **Resilient Tool Caching & Error Safety:** A disk-backed cache (`.tool_cache/`) wrapping DuckDuckGo (`ddgs`) search that prevents redundant API requests, truncates large context payloads, and skips caching on API errors or blank results.
- **Node-Level Corrective Retries:** Catches `tool_call_invalid` syntax errors during multi-turn LLM reasoning and applies node-level retries with instruction injection before execution breaks.
- **URL Normalization & Honesty Metrics:** Evaluates citation honesty ratios ($N_{\text{valid}} / N_{\text{cited}}$) by extracting and normalizing URLs from markdown reports against retrieved tool payloads.
- **Stream-Based Metrics Wrapper:** Streams state transitions to capture wall-clock latency, total token consumption (input/output), tool invocation counts, and explicit failure reasons (`rate_limit`, `tool_call_invalid`, `recursion`, `empty_report`).

---

##  Repository Structure

```text
research-team-agents/
├── .results/           # Local JSON execution logs & run payloads (ignored by git)
├── .tool_cache/        # Local disk cache for DuckDuckGo search queries (ignored by git)
├── config.py           # Centralized LLM setup & TokenTracker utility
├── tools.py            # Custom web search tool with caching, truncation & retries
├── system_a_baseline.py# Declarative ReAct Graph Agent implementation & metrics wrapper
├── benchmark_groq.py   # Task 2 Groq model comparison script (20-run benchmark)
├── main.py             # Early manual ReAct loop & structured output router benchmark
├── NOTES.md            # Empirical evaluation findings, failure audits & architectural weaknesses
├── requirements.txt    # Python project dependencies
└── .gitignore          # Prevents tracking of .venv, .env, .results/, and .tool_cache/