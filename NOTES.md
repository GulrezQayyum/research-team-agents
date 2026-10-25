# System A Baseline Evaluation
## Overview
System A implements a basic ReAct loop structure in LangGraph. Benchmarking reveals key trade-offs between prompting rules, error recovery, and graph edge logic.

## Top 3 Structural Weaknesses

### 1. Off-By-One Tool Loop Edge Routing
* **Mechanic:** Routing on `AIMessage.tool_calls` prior to tool execution truncates the graph turn count prematurely. 
* **Impact:** Without counting executed `ToolMessage` payloads or routing through a dedicated tool-free synthesis node, the graph exits while an `AIMessage` containing tool calls is still pending. This causes silent execution failures (producing 0-word reports).

### 2. Search Provider Instability & Rate-Limit Thrashing
* **Mechanic:** Unauthenticated HTML scraping (`ddgs`) experiences connection timeouts and rate limits under sequential multi-turn loops.
* **Impact:** The LLM receives empty search string responses, misinterprets API errors as a lack of web content, and exhausts its iteration budget on repetitive variations of the same query.

### 3. Token Accumulation & Citation Hallucination
* **Mechanic:** Search results containing unstructured body text are stored linearly in conversation history (`messages`).
* **Impact:** Prompt input tokens grow non-linearly over multiple turns. When tool results omit full link paths, the LLM hallucinates plausible URLs to meet prompt constraints.