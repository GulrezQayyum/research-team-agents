# NOTES.md — System A Baseline Evaluation

## Top 3 Observed Weaknesses

1. **Unbounded Loop Vulnerability & Soft Limit Failures**
   Prompt-based constraints ("AT MOST 4 searches") fail when search results return empty or fail. The LLM continuously retries queries, leading to potential `RecursionError` crashes unless capped in the graph routing edge.

2. **Silent Scraper Throttling & Rate-Limit Errors**
   Unauthenticated HTML web searches (`ddgs`) trigger site timeouts and empty responses under back-to-back execution, causing the agent to hallucinate queries or return lower-quality reports.

3. **Context Accumulation & Citation Hallucination**
   Appending raw search payloads into `messages` inflates prompt tokens exponentially. When returned search payloads omit full URLs, the model fabricates plausible URL endpoints to satisfy system prompt rules.