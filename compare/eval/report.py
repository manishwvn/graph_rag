"""Generate the markdown comparison report from metrics.json.

Every line is derived from the metrics file. The previous version printed a
fixed narrative ("Graph excels with hops=2", "handles abstention well") next to
tables that showed the opposite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS = "compare/eval/metrics.json"
BUILD_STATS = "compare/eval/build_stats.json"
REPORT = "compare/comparison_report.md"

LABELS = {"vector": "Vector (k={k})", "graph_hops1": "Graph hops=1", "graph_hops2": "Graph hops=2"}


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def num(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"


def _label(name: str, k: int) -> str:
    return LABELS.get(name, name).format(k=k)


def _winner(summary: dict, systems: list[str], key: str, higher_is_better: bool = True) -> str:
    vals = [(s, summary[s].get(key, 0.0)) for s in systems]
    best = (max if higher_is_better else min)(vals, key=lambda x: x[1])
    tied = [s for s, v in vals if abs(v - best[1]) < 1e-9]
    return ", ".join(tied)


def generate_report(metrics_path: str = METRICS, output_path: str = REPORT, build_stats_path: str = BUILD_STATS) -> str:
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    cfg: dict[str, Any] = data.get("config", {})
    summary: dict[str, dict] = data["summary"]
    details: dict[str, list[dict]] = data["details"]
    k = cfg.get("k", 4)
    systems = list(summary)

    L: list[str] = []
    L.append("# Vector vs Graph RAG — Comparison Report\n")
    L.append(f"Generated from `{metrics_path}`. Every number below is read from that file.\n")

    L.append("## Setup\n")
    L.append("| Setting | Value |")
    L.append("|---|---|")
    L.append(f"| Corpus chunks | {cfg.get('vector_chunks', '?')} (chunk_size={cfg.get('chunk_size')}, overlap={cfg.get('chunk_overlap')}) |")
    g = cfg.get("graph", {})
    L.append(f"| Graph | {g.get('nodes', '?')} nodes, {g.get('edges', '?')} edges, provenance over {g.get('chunks', '?')} chunks |")
    L.append(f"| Retrieval budget | k={k} chunks per query, max_triples={cfg.get('max_triples')} |")
    L.append(f"| Generation model | `{cfg.get('model')}` (identical prompt for both systems) |")
    L.append(f"| Embedding model | `{cfg.get('embed_model')}` |")
    L.append(f"| Queries | {cfg.get('queries')} ({summary[systems[0]]['answerable_count']} answerable, {summary[systems[0]]['negative_count']} negative) |")
    L.append("")
    L.append(
        "Both systems return a ranked list of `chunk_uid`s, so retrieval metrics are "
        "measured in one id space; both are given the same generation model, the same "
        "prompt and the same context budget (see *Avg context chars*). Retrieval metrics "
        "average over answerable queries only; the negative queries are scored on "
        "abstention instead. Embedding cost is reported in the build table below "
        "rather than per query, because query embeddings are cached and a repeat "
        "run would bill zero.\n"
    )

    L.append("## Summary\n")
    L.append("| Metric | " + " | ".join(_label(s, k) for s in systems) + " | Best |")
    L.append("|---|" + "---|" * (len(systems) + 1))
    rows = [
        (f"Hit@{k}", f"hit@{k}", pct, True),
        (f"Precision@{k}", f"precision@{k}", pct, True),
        (f"Recall@{k}", f"recall@{k}", pct, True),
        ("MRR", "mrr", lambda v: num(v, 3), True),
        ("Answer keyword recall", "keyword_recall", pct, True),
        ("Judge: correct (answerable)", "judge_correct_answerable", pct, True),
        ("Judge: grounded (all)", "judge_grounded", pct, True),
        ("Judge coverage (rows scored)", "judge_coverage", pct, True),
        ("Abstained on answerable", "abstention_on_answerable", pct, False),
        ("Abstained on negative", "abstention_on_negative", pct, True),
        ("Avg context chars", "context_chars", lambda v: f"{v:.0f}", True),
        ("Avg latency (s)", "latency_s", lambda v: num(v, 2), False),
        ("Avg Groq tokens/query", "llm_tokens", lambda v: f"{v:.0f}", False),
    ]
    for title, key, fmt, hib in rows:
        L.append(
            f"| {title} | " + " | ".join(fmt(summary[s].get(key, 0.0)) for s in systems) + f" | {_winner(summary, systems, key, hib)} |"
        )
    L.append("")

    # ---- build cost
    try:
        bs = json.loads(Path(build_stats_path).read_text(encoding="utf-8"))
    except Exception:
        bs = {}
    if bs:
        L.append("## Build cost\n")
        L.append("| | Vector | Graph |")
        L.append("|---|---|---|")
        v, gr = bs.get("vector", {}), bs.get("graph", {})
        L.append(f"| Chunks indexed | {v.get('chunks_indexed', '?')}/{v.get('chunks_total', '?')} | {gr.get('chunks_indexed', '?')}/{gr.get('chunks_total', '?')} |")
        L.append(f"| API calls | {v.get('api_calls', '?')} (NVIDIA embed) | {gr.get('api_calls', '?')} (Groq extract) |")
        L.append(f"| Wall clock | {v.get('total_time_s', '?')}s | {gr.get('total_time_s', '?')}s |")
        L.append(f"| Index | {v.get('chroma', {}).get('size_mb', '?')} MB ChromaDB | {gr.get('nodes', '?')} nodes / {gr.get('edges', '?')} edges JSON |")
        L.append(f"| Canonicalization | — | {gr.get('aliases_merged', '?')} aliases merged ({gr.get('nodes_before_canonicalization', '?')} → {gr.get('nodes', '?')} nodes) |")
        L.append("")

    # ---- per query type
    L.append("## By query type\n")
    types: list[str] = []
    for r in details[systems[0]]:
        if r["type"] not in types:
            types.append(r["type"])
    L.append("| Type | n | Metric | " + " | ".join(_label(s, k) for s in systems) + " |")
    L.append("|---|---|---|" + "---|" * len(systems))
    for t in types:
        subset = {s: [r for r in details[s] if r["type"] == t] for s in systems}
        n = len(subset[systems[0]])
        answerable = subset[systems[0]][0]["answerable"] if n else False
        metric_rows = ([(f"hit@{k}", f"hit@{k}", pct)] if answerable else []) + [
            ("judge correct", "judge_correct", pct),
            ("judge grounded", "judge_grounded", pct),
        ]
        for i, (title, key, fmt) in enumerate(metric_rows):
            head = f"| {t} | {n} | " if i == 0 else "|  |  | "
            def cell(system: str, key: str = key, fmt=fmt, subset=subset) -> str:
                vals = [r[key] for r in subset[system] if isinstance(r.get(key), (int, float))]
                return fmt(sum(vals) / len(vals)) if vals else "n/a"

            L.append(head + f"{title} | " + " | ".join(cell(s) for s in systems) + " |")
    L.append("")

    # ---- per query
    L.append("## Per query\n")
    L.append("| id | type | " + " | ".join(f"{_label(s, k)} hit@{k} / correct" for s in systems) + " |")
    L.append("|---|---|" + "---|" * len(systems))
    for i, base in enumerate(details[systems[0]]):
        cells = []
        for s in systems:
            r = details[s][i]
            h = pct(r[f"hit@{k}"]) if r["answerable"] else "n/a"
            mark = "—" if r["judge_correct"] is None else ("✅" if r["judge_correct"] else "❌")
            cells.append(f"{h} / {mark}")
        L.append(f"| {base['id']} | {base['type']} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- disagreements, derived
    L.append("## Where the systems disagree\n")
    disagreements = []
    for i, base in enumerate(details[systems[0]]):
        verdicts = {s: details[s][i]["judge_correct"] for s in systems}
        if None in verdicts.values():
            continue  # an unscored judgement is not a disagreement
        if len(set(verdicts.values())) > 1:
            won = [s for s, v in verdicts.items() if v]
            lost = [s for s, v in verdicts.items() if not v]
            reason = next(details[s][i]["judge_reason"] for s in lost)
            disagreements.append(f"- **{base['id']}** ({base['type']}) — correct: {', '.join(won)}; wrong: {', '.join(lost)}. Judge on {lost[0]}: _{reason}_")
    L.extend(disagreements or ["- No query separated the systems on judged correctness."])
    L.append("")

    # ---- conclusions, derived
    L.append("## Findings\n")
    for title, key, fmt, hib in rows:
        best = _winner(summary, systems, key, hib)
        vals = ", ".join(f"{_label(s, k)} {fmt(summary[s].get(key, 0.0))}" for s in systems)
        L.append(f"- **{title}** — best: {best} ({vals})")
    L.append("")
    h1, h2 = "graph_hops1", "graph_hops2"
    if h1 in summary and h2 in summary:
        d = summary[h2].get(f"recall@{k}", 0) - summary[h1].get(f"recall@{k}", 0)
        direction = "improves" if d > 0 else ("degrades" if d < 0 else "does not change")
        L.append(
            f"- Going from hops=1 to hops=2 {direction} recall@{k} by {d * 100:+.1f} points "
            f"({pct(summary[h1][f'recall@{k}'])} → {pct(summary[h2][f'recall@{k}'])}) at "
            f"{num(summary[h2]['latency_s'] - summary[h1]['latency_s'], 2)}s extra latency."
        )
    L.append("")

    report = "\n".join(L)
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"[report] written to {output_path}")
    return report
