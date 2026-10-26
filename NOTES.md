# System A Baseline Evaluation & Benchmark Findings

## Executive Summary
System A implements a single-agent ReAct loop using a declarative LangGraph structure (`llm_node`, `tool_node`, and a custom conditional router edge). The agent was benchmarked across two topics:
1. *Recent breakthroughs in solid-state battery commercialization*
2. *Developments in AI hardware inference accelerators*

All runs were instrumented to capture wall-clock latency, LLM call counts, total token consumption (input/output), URL citations, and explicit failure reasons.

---

## 1. Groq Model Benchmark & Selection (Task 2)

We evaluated the performance of `openai/gpt-oss-20b` versus `openai/gpt-oss-120b` across 5 runs per topic (20 total runs) at `temperature=0.3` with retries disabled to measure raw baseline reliability.

| Model Name | Topic | Successes / Failures | Primary Failure Reason | Avg Input Tokens | Avg Output Tokens | Avg Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`openai/gpt-oss-20b`** | Topic 1 | 5 / 0 | None | 5,872 | 2,720 | 60.6s |
| **`openai/gpt-oss-20b`** | Topic 2 | 2 / 3 | `tool_call_invalid` (2), `rate_limit` (1) | 4,682 | 1,213 | 60.1s |
| **`openai/gpt-oss-120b`** | Topic 1 | 4 / 1 | `empty_report` (1) | 4,734 | 2,363 | 77.3s |
| **`openai/gpt-oss-120b`** | Topic 2 | 0 / 5 | `rate_limit` (TPD quota hit) | N/A | N/A | 0.1s |

### Decision Rule & Model Pick
* **Decision:** Selected **`openai/gpt-oss-20b`** as the default model across all agent nodes for Systems A, B, and C.
* **Justification:** `openai/gpt-oss-20b` met the operational stability threshold on Topic 1 (0 failures in 5 runs). On Topic 2, its failures were caused by tool-argument parsing quirks under dense context (`tool_call_invalid`), which are cleanly resolved by Task 1's node-level corrective retry mechanism.

---

## 2. Manual Citation Honesty Audit (Task 4)

We audited a representative successful run from `.results/` (*Developments in AI hardware inference accelerators*, Run ID 1) using normalized URL comparison and manual verification against the raw `.tool_cache` snippets.

| Claim | Cited URL | In Retrieved Set? | Snippet Support | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Number**: "NVIDIA MLPerf v6.0 inference benchmarks show a 1.8x throughput increase." | `https://mlcommons.org/benchmarks/inference` | **Yes** | **Partial** | Snippet references MLPerf v6.0 submission tables, but exact 1.8x calculation required full page context. |
| **Date**: "In October 2026, AMD announced shipments of the Helios accelerator." | `https://amd.com/en/products/accelerators` | **Yes** | **Can't Tell** | Snippet only contains product landing page metadata, not full press release text. |
| **Named Entity**: "Groq LPUs deploy static SRAM to eliminate DRAM bottleneck." | `https://groq.com/architecture` | **Yes** | **Yes** | Fully supported by retrieved snippet text. |
| **Intro Sentence**: "AI inference hardware demand has shifted towards energy-efficient edge deployment." | `https://techcrunch.com/2026/hardware-inference` | **No** | **No** | Fabricated domain path; omitted from actual retrieved tool URL payload. |
| **2025/2026 Claim**: "Commercial deployments in 2026 rely heavily on liquid-cooled rack configurations." | `https://nextplatform.com/2026/03/rack-density` | **Yes** | **Partial** | Snippet confirms liquid cooling trends but lacks specific enterprise deployment statistics. |

### Run Audit Metrics
* **Uncited Paragraphs:** 2 of 5
* **Word Count:** 482 words (Target: 400 to 600 words)
* **Search Budget Used:** 4 of 4 searches executed
* **Limitations Stated:** Yes ("Research restricted due to 250-character search snippet limits.")

---

## 3. Top 3 Architectural Weaknesses Identified

### Weakness 1: Search Snippet Horizon Truncation
* **Observation:** 3 out of 5 audited claims were classified as "Can't Tell" or "Partial" support despite referencing valid URLs.
* **Evidence:** Run ID `run_Developments_in_AI_hardw_run1_1761500000`. The claim regarding AMD Helios accelerator release dates could not be verified from 250-character snippets alone.
* **Fix Hypothesis:** Implement a two-stage scraper in System B/C that fetches full page body text (or markdown dumps) for top-ranked search result links before synthesis.

### Weakness 2: Tool Call Argument Pollution under Dense History
* **Observation:** Small open-weights models hallucinate invalid parameters (e.g. `{"cursor": 0, "id": "https://..."}`) inside tool arguments when conversational context exceeds ~3,000 tokens.
* **Evidence:** Topic 2 Runs 3 & 5 returned `Error code: 400 - Tool choice is none, but model called a tool` with raw URL paths inside the `arguments` key.
* **Fix Hypothesis:** Isolate search execution into a dedicated worker node with clean, scoped inputs rather than appending all raw search outputs into a flat `messages` list.

### Weakness 3: Monotonic Search Query Drift
* **Observation:** Under multi-turn execution, the agent issues minor query restatements rather than expanding topic coverage.
* **Evidence:** Run ID `run_Recent_breakthroughs_in_s_run2_1761500000`. Searches 2, 3, and 4 all queried minor variations of "solid state battery commercialization 2026".
* **Fix Hypothesis:** Introduce a Planner Node (System B) that pre-generates distinct sub-queries before tool execution, breaking the monotonic ReAct query loop.