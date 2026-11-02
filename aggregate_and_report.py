"""Merge blind quality scores and produce the Stage 8 report."""

import json
from pathlib import Path

import pandas as pd


def aggregate_benchmark_results(
    csv_path: str = "eval_results/metrics_summary.csv",
    scores_path: str = "eval_results/blind_scores.json",
    key_path: str = "eval_results/scoring_key_DO_NOT_OPEN.json",
) -> None:
    df = pd.read_csv(csv_path)
    df["quality_score"] = float("nan")

    if Path(scores_path).exists() and Path(key_path).exists():
        blind_scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
        key_mapping = json.loads(Path(key_path).read_text(encoding="utf-8"))
        score_by_filename = {
            key_mapping[sample_id]: score
            for sample_id, score in blind_scores.items()
            if sample_id in key_mapping and score is not None
        }
        df["quality_score"] = df.apply(
            lambda row: score_by_filename.get(
                f"{row['topic_id']}_{row['system_name']}_run{int(row['run_number'])}.txt"
            ),
            axis=1,
        )
    else:
        print("Note: blind_scores.json is missing; quality scores remain unscored.")

    summary = df.groupby("system_name").agg(
        wall_time_sec_mean=("wall_time_sec", "mean"),
        wall_time_sec_std=("wall_time_sec", "std"),
        total_tokens_mean=("total_tokens", "mean"),
        total_tokens_std=("total_tokens", "std"),
        llm_calls_mean=("llm_calls", "mean"),
        llm_calls_std=("llm_calls", "std"),
        tool_calls_mean=("tool_calls", "mean"),
        tool_calls_std=("tool_calls", "std"),
        citation_validity_mean=("citation_validity", "mean"),
        citation_validity_std=("citation_validity", "std"),
        failure_rate=("failed", "mean"),
        runs=("failed", "count"),
        quality_score_mean=("quality_score", "mean"),
        quality_score_std=("quality_score", "std"),
    )
    summary["failure_rate"] *= 100
    summary["tokens_per_quality_point"] = (
        summary["total_tokens_mean"] / summary["quality_score_mean"]
    )

    output_dir = Path(csv_path).parent
    summary.to_csv(output_dir / "aggregated_report.csv")
    lines = [
        "# Stage 8 evaluation analysis",
        "",
        "Means and standard deviations are calculated across all five fixed topics and three runs per system.",
        "Citation validity is the share of unique cited URLs present in that run's search results.",
        "",
        "## System comparison",
    ]
    for system_name, row in summary.iterrows():
        quality = row["quality_score_mean"]
        quality_text = "not scored" if pd.isna(quality) else f"{quality:.2f}/5"
        cost_text = "n/a" if pd.isna(row["tokens_per_quality_point"]) else f"{row['tokens_per_quality_point']:.0f}"
        lines.append(
            f"- **{system_name}**: quality {quality_text}, "
            f"{row['total_tokens_mean']:.0f} tokens/run, "
            f"{row['wall_time_sec_mean']:.2f}s/run, "
            f"{row['citation_validity_mean']:.2f} citation validity, "
            f"{row['failure_rate']:.1f}% failure rate, "
            f"{cost_text} tokens per quality point."
        )
    lines += [
        "",
        "## Failure modes",
        "Use `metrics_summary.csv` for run-level failure reasons, including crashes, limit exits, and empty reports.",
        "",
        "## Checkpoint",
        "Use System A for lower-cost, direct research when a bounded single-agent pass is sufficient. "
        "Use System B when a fixed research/write/critic workflow is worth its predictable extra cost. "
        "Use System C when adaptive routing may justify the highest cost and latency.",
    ]
    (output_dir / "evaluation_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string())
    print(f"Saved {output_dir / 'aggregated_report.csv'} and {output_dir / 'evaluation_analysis.md'}")


if __name__ == "__main__":
    aggregate_benchmark_results()
