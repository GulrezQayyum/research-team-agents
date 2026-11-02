#!/usr/bin/env python3
"""Evaluation checks for the Research Team project.

Run from the project root:
  python inspect_runs.py failures
  python inspect_runs.py handoff <run_json_file>
  python inspect_runs.py routes  <run_json_file>
  python inspect_runs.py blind
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("eval_results")
URL = re.compile(r"https?://[^\s)\]>\"'<,]+")


def find_csv():
    for p in [ROOT / "metrics_summary.csv", Path("metrics_summary.csv")]:
        if p.exists():
            return p
    hits = list(Path(".").rglob("metrics_summary.csv"))
    if hits:
        return hits[0]
    sys.exit("metrics_summary.csv not found")


def load_rows():
    with open(find_csv(), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify(msg):
    m = (msg or "").lower()
    if m in ("", "none"):
        return "none"
    if "tool_use_failed" in m or "tool choice is none" in m:
        return "tool_call_invalid (final call)"
    if "rate_limit" in m or "429" in m:
        return "rate_limit (quota, not a real run)"
    if "recursion limit" in m:
        return "recursion_limit"
    if "without a valid url" in m:
        return "researcher_no_valid_url"
    if "invalid json" in m:
        return "critic_invalid_json"
    return "other"


def num(x):
    return float(x) if x not in (None, "") else 0.0


def cmd_failures():
    rows = load_rows()
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system_name"]].append(r)
    for system, rs in sorted(by_sys.items()):
        failed = [r for r in rs if r["failed"] == "True"]
        ok = [r for r in rs if r["failed"] != "True"]
        print(f"\n{system}: {len(ok)}/{len(rs)} completed")
        cats = defaultdict(lambda: [0, 0.0, 0.0])
        for r in failed:
            c = cats[classify(r["error_message"])]
            c[0] += 1
            c[1] += num(r["total_tokens"])
            c[2] += num(r["wall_time_sec"])
        for name, (n, tok, sec) in cats.items():
            print(f"  {n}x {name}: {tok:.0f} tokens, {sec:.0f}s")
        by_topic = defaultdict(lambda: [0, 0])
        for r in rs:
            by_topic[r["topic_id"]][1] += 1
            if r["failed"] != "True":
                by_topic[r["topic_id"]][0] += 1
        print("  completed per topic:",
              ", ".join(f"T{t.split('_')[1]} {a}/{b}" for t, (a, b) in sorted(by_topic.items())))
        if ok:
            n = len(ok)
            tok = sum(num(r["total_tokens"]) for r in ok) / n
            sec = sum(num(r["wall_time_sec"]) for r in ok) / n
            cited = sum(num(r["cited_urls"]) for r in ok)
            valid = sum(num(r["valid_citations"]) for r in ok)
            zero = sum(1 for r in ok if num(r["cited_urls"]) == 0)
            print(f"  completed means: {tok:.0f} tokens, {sec:.1f}s")
            print(f"  URLs cited: {cited:.0f} ({valid:.0f} valid); reports with zero URLs: {zero}/{n}")


def norm(u):
    return u.rstrip(".,;)/").lower()


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, o


def text_for(data, needles, skip=()):
    parts = []
    for p, v in walk(data):
        pl = p.lower()
        if any(n in pl for n in needles) and not any(s in pl for s in skip):
            parts.append(str(v))
    return "\n".join(parts)


def urls_in(text):
    return {norm(u) for u in URL.findall(text)}


def cmd_handoff(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    print("top-level keys:", list(data) if isinstance(data, dict) else type(data))
    notes = text_for(data, ("note", "finding"))
    draft = text_for(data, ("draft",))
    report = text_for(data, ("report",), skip=("citation",))
    n_urls, d_urls, r_urls = urls_in(notes), urls_in(draft), urls_in(report)
    print(f"notes : {len(notes)} chars, {len(n_urls)} URLs")
    print(f"draft : {len(draft)} chars, {len(d_urls)} URLs")
    print(f"report: {len(report)} chars, {len(r_urls)} URLs")
    if not notes:
        print("No notes found in this file. Save the state in the run payload first.")
        return
    final = d_urls | r_urls
    print(f"note URLs that reached the final text: {len(n_urls & final)}/{len(n_urls)}")
    print("URLs in final text but not in notes:", sorted(final - n_urls) or "none")
    for u in sorted(n_urls - final)[:10]:
        print("  lost:", u)
    markers = re.findall(r"\[\d+\]", report or draft)
    print(f"numeric markers like [1] in final text: {len(markers)}")


def cmd_routes(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    hits = [(p, v) for p, v in walk(data)
            if any(k in p.lower() for k in ("route", "decision", "supervisor", "next"))]
    if not hits:
        print("No route data saved in this file. Add route_log to the state and rerun.")
        return
    for p, v in hits:
        print(f"{p}: {v}")


def index_reports():
    idx = {}
    for p in ROOT.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict):
            s, t = d.get("system_name"), d.get("topic_id")
            r = d.get("run_number", d.get("run_id"))
            if s and t and r is not None:
                idx[(s, t, int(r))] = d
    return idx


def report_text(d):
    for k in ("report", "final_report", "draft"):
        if d.get(k):
            return str(d[k])
    return ""


def cmd_blind():
    rows = load_rows()
    idx = index_reports()
    if not idx:
        print("No run JSON files with system_name, topic_id and run_number under eval_results/.")
        print("Paste me `ls eval_results/reports | head` and the keys of one file.")
        return
    chosen = {}
    for r in sorted(rows, key=lambda r: (r["system_name"], r["topic_id"], int(r["run_number"]))):
        key = (r["system_name"], r["topic_id"])
        if r["failed"] != "True" and key not in chosen:
            chosen[key] = int(r["run_number"])
    items = []
    for (system, topic), run in chosen.items():
        d = idx.get((system, topic, run))
        text = report_text(d) if d else ""
        if text:
            items.append((system, topic, run, text))
        else:
            print(f"no report text for {system} {topic} run {run}")
    random.seed(7)
    random.shuffle(items)
    key_map, packet = {}, []
    for i, (system, topic, run, text) in enumerate(items, 1):
        rid = f"R{i:02d}"
        key_map[rid] = {"system": system, "topic": topic, "run": run}
        packet.append(f"## {rid}\n\n(topic: {topic.split('_', 2)[-1]})\n\n{text}\n\n---\n")
    Path("blind_packet.md").write_text("\n".join(packet), encoding="utf-8")
    Path("blind_key.json").write_text(json.dumps(key_map, indent=2), encoding="utf-8")
    with open("blind_scores_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "coverage", "accuracy_vs_sources", "structure", "usefulness"])
        for rid in key_map:
            w.writerow([rid, "", "", "", ""])
    print(f"{len(items)} reports -> blind_packet.md; scores -> blind_scores_template.csv")
    print("Do not open blind_key.json until every score is filled in.")


if __name__ == "__main__":
    cmds = {"failures": cmd_failures, "handoff": cmd_handoff,
            "routes": cmd_routes, "blind": cmd_blind}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](*sys.argv[2:])