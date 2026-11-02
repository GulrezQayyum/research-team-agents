"""Stage 8 benchmark runner and blind-scoring preparation.

The benchmark intentionally keeps the five topics in this module and writes
them once to ``benchmark_topics.json``.  Quality scores are entered separately
for the anonymized reports and merged by ``aggregate_and_report.py``.
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


TOPICS = [
    {"id": "topic_1_factual", "prompt": "Explain the mechanics of photosystem II in oxygenic photosynthesis."},
    {"id": "topic_2_overview", "prompt": "Provide a comprehensive overview of quantum computing architectures in 2026."},
    {"id": "topic_3_conflicting", "prompt": "Analyze conflicting evidence regarding caffeine consumption and cardiovascular outcomes."},
    {"id": "topic_4_recent", "prompt": "Summarize key developments in global semiconductor trade policies over the past year."},
    {"id": "topic_5_niche", "prompt": "Trace the development and dialectal features of the Elfdalian language."},
]
RUNS_PER_TOPIC = 3
URL_PATTERN = re.compile(r"https?://[^\s><\"'()\]}]+")


@dataclass
class RunMetrics:
    topic_id: str
    system_name: str
    run_number: int
    wall_time_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
    cited_urls: int
    valid_citations: int
    citation_validity: float
    failed: bool
    error_message: str = ""


def _urls(value: Any) -> set[str]:
    return {url.rstrip(".,;!?:'\"]}") for url in URL_PATTERN.findall(str(value))}


def calculate_citation_validity(report_text: str, source_urls: list[str] | set[str]) -> tuple[int, int, float]:
    cited = _urls(report_text)
    sources = _urls(source_urls)
    valid = cited & sources
    return len(cited), len(valid), len(valid) / len(cited) if cited else 0.0


class AgentEvaluator:
    def __init__(self, output_dir: str = "eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_topics(self) -> None:
        path = self.output_dir / "benchmark_topics.json"
        if not path.exists():
            path.write_text(json.dumps(TOPICS, indent=2) + "\n", encoding="utf-8")

    def run_benchmark(
        self,
        system_fn: Callable[[str], dict[str, Any]],
        system_name: str,
        topic: dict[str, str],
        run_number: int,
    ) -> RunMetrics:
        started = time.perf_counter()
        report = ""
        result: dict[str, Any] = {}
        error = ""
        try:
            result = system_fn(topic["prompt"])
            report = str(result.get("report", ""))
        except Exception as exc:
            error = str(exc)

        cited, valid, validity = calculate_citation_validity(
            report, result.get("source_urls", [])
        )
        failed = bool(error or result.get("failed", False) or len(report.strip()) == 0)
        if not error:
            error = str(result.get("error_message", ""))
        if not report.strip() and not error:
            error = "empty_report"

        metrics = RunMetrics(
            topic_id=topic["id"],
            system_name=system_name,
            run_number=run_number,
            wall_time_sec=round(time.perf_counter() - started, 3),
            prompt_tokens=int(result.get("prompt_tokens", 0)),
            completion_tokens=int(result.get("completion_tokens", 0)),
            total_tokens=int(result.get("prompt_tokens", 0)) + int(result.get("completion_tokens", 0)),
            llm_calls=int(result.get("llm_calls", 0)),
            tool_calls=int(result.get("tool_calls", 0)),
            cited_urls=cited,
            valid_citations=valid,
            citation_validity=round(validity, 4),
            failed=failed,
            error_message=error,
        )
        if report:
            filename = f"{topic['id']}_{system_name}_run{run_number}.txt"
            (self.reports_dir / filename).write_text(report, encoding="utf-8")
        return metrics


def run_system_a(prompt: str) -> dict[str, Any]:
    from system_a_baseline import run_agent_with_metrics

    result = run_agent_with_metrics(prompt, 0)
    return {
        "report": result.get("report", ""),
        "prompt_tokens": result.get("input_tokens", 0),
        "completion_tokens": result.get("output_tokens", 0),
        "llm_calls": result.get("llm_calls", 0),
        "tool_calls": result.get("tool_calls", 0),
        "source_urls": result.get("retrieved_urls", []),
        "failed": result.get("status_failed", False),
        "error_message": result.get("error_message") or result.get("failure_reason", ""),
    }


def run_system_b(prompt: str) -> dict[str, Any]:
    from config import tracker
    from system_b_fixed_pipeline import run_and_stream_pipeline

    before_in, before_out = tracker.input_tokens, tracker.output_tokens
    state = run_and_stream_pipeline(prompt)
    prompt_tokens = tracker.input_tokens - before_in
    completion_tokens = tracker.output_tokens - before_out
    notes = state["research_notes"]
    return {
        "report": state["draft"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": 3 + state["revisions"],
        "tool_calls": sum(str(note).count("Link: http") for note in notes),
        "source_urls": sorted(set().union(*(_urls(note) for note in notes))),
        "failed": not state["draft"],
        "error_message": "",
    }


def run_system_c(prompt: str) -> dict[str, Any]:
    from config import tracker
    from system_c_supervisor import run_supervisor_pipeline

    before_in, before_out = tracker.input_tokens, tracker.output_tokens
    state = run_supervisor_pipeline(prompt)
    notes = state["research_notes"]
    return {
        "report": state["draft"],
        "prompt_tokens": tracker.input_tokens - before_in,
        "completion_tokens": tracker.output_tokens - before_out,
        "llm_calls": len(state["route_sequence"]) + len(state["decision_log"]),
        "tool_calls": sum(str(note).count("Link: http") for note in notes),
        "source_urls": sorted(set().union(*(_urls(note) for note in notes))),
        "failed": state["failed"] or not state["draft"],
        "error_message": "; ".join(state["errors"]),
    }


def prepare_blind_scoring(evaluator: AgentEvaluator, seed: int = 8) -> None:
    blind_dir = evaluator.output_dir / "blind_evaluation"
    blind_dir.mkdir(exist_ok=True)
    files = sorted(evaluator.reports_dir.glob("*.txt"))
    random.Random(seed).shuffle(files)
    mapping: dict[str, str] = {}
    for index, source in enumerate(files, 1):
        sample_id = f"Sample_{index:03d}"
        mapping[sample_id] = source.name
        (blind_dir / f"{sample_id}.txt").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (evaluator.output_dir / "scoring_key_DO_NOT_OPEN.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    (evaluator.output_dir / "blind_scores_template.json").write_text(
        json.dumps({sample_id: None for sample_id in mapping}, indent=2) + "\n",
        encoding="utf-8",
    )


def run_benchmark() -> None:
    evaluator = AgentEvaluator()
    evaluator.save_topics()
    systems = {"System_A": run_system_a, "System_B": run_system_b, "System_C": run_system_c}
    rows: list[dict[str, Any]] = []
    for system_name, runner in systems.items():
        for topic in TOPICS:
            for run_number in range(1, RUNS_PER_TOPIC + 1):
                print(f"Running {system_name} | {topic['id']} | {run_number}/{RUNS_PER_TOPIC}")
                rows.append(asdict(evaluator.run_benchmark(runner, system_name, topic, run_number)))
    with (evaluator.output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prepare_blind_scoring(evaluator)
    print("Benchmark telemetry and blind-scoring files are ready.")


if __name__ == "__main__":
    run_benchmark()
