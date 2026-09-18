"""Aggregation, uncertainty and statistical testing.

Input : ``<run_dir>/runs.jsonl`` (one record per system x repeat x claim).
Output: CSV tables, a Markdown summary and figures in ``<run_dir>/analysis``.

Statistics
----------
* Point estimate: mean over claims of the per-claim value averaged over repeats.
* 95% CI: percentile bootstrap over claims (B = 10 000, seed 42).
* Paired comparison vs the dense baseline: paired bootstrap CI of the mean difference,
  Wilcoxon signed-rank test (continuous metrics) and exact McNemar test (verdict correctness,
  pooled over repeats); Holm correction across systems within each metric.
* Generation stability: SD across repeats of the system-level mean.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

RETRIEVAL_METRICS = ["doc_precision@k", "doc_recall@k", "doc_ndcg@k", "doc_mrr",
                     "sent_precision@k", "sent_recall@k", "sent_ndcg@k", "sent_mrr"]
EVIDENCE_METRICS = ["evidence_precision", "evidence_recall", "evidence_f1", "full_rationale_recovered"]
VERDICT_METRICS = ["verdict_correct", "fever_score", "json_valid"]
ALL_METRICS = RETRIEVAL_METRICS + EVIDENCE_METRICS + VERDICT_METRICS
BASELINE = "Dense"


def load_runs(run_dir: str) -> pd.DataFrame:
    rows = []
    with open(os.path.join(run_dir, "runs.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            flat = {k: v for k, v in r.items() if k not in ("metrics", "llm", "diag", "timing")}
            flat.update(r["metrics"])
            flat.update({f"t_{k}": v for k, v in r["timing"].items()})
            flat.update({k: v for k, v in r["llm"].items() if k != "calls_by_stage"})
            flat["calls_by_stage"] = json.dumps(r["llm"].get("calls_by_stage", {}))
            flat["diag"] = json.dumps(r.get("diag", {}))
            rows.append(flat)
    return pd.DataFrame(rows)


def bootstrap_ci(x: np.ndarray, B: int = 10000, seed: int = 42) -> tuple:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_bootstrap(d: np.ndarray, B: int = 10000, seed: int = 42) -> tuple:
    return bootstrap_ci(d, B, seed)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted((p, k) for k, p in pvals.items() if not math.isnan(p))
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k in pvals:
        out.setdefault(k, float("nan"))
    return out


def per_claim_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average each metric over repeats -> one row per (system, qid)."""
    return df.groupby(["system", "qid"], as_index=False)[ALL_METRICS].mean()


def main_table(df: pd.DataFrame, systems: List[str]) -> pd.DataFrame:
    pc = per_claim_table(df)
    rows = []
    for s in systems:
        sub = pc[pc.system == s]
        row = {"system": s, "n_claims": int(sub.qid.nunique()), "n_repeats": int(df[df.system == s].repeat.nunique())}
        for m in ALL_METRICS:
            x = sub[m].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x)
            row[m] = float(np.nanmean(x)) if np.any(~np.isnan(x)) else float("nan")
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = lo, hi
            # stability across repeats (SD of the system-level mean)
            per_rep = df[df.system == s].groupby("repeat")[m].mean(numeric_only=True)
            row[f"{m}_sd_repeats"] = float(per_rep.std(ddof=0)) if len(per_rep) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def verdict_table(df: pd.DataFrame, systems: List[str], labels: List[str]) -> tuple:
    rows, cms = [], {}
    for s in systems:
        sub = df[df.system == s]
        y, p = sub.gold_label.tolist(), sub.pred_label.tolist()
        cm = pd.DataFrame(0, index=labels, columns=labels + ["INVALID"])
        for a, b in zip(y, p):
            if a in labels:
                cm.loc[a, b if b in cm.columns else "INVALID"] += 1
        cms[s] = cm
        row = {"system": s, "n": len(y), "accuracy": float(np.mean([a == b for a, b in zip(y, p)]))}
        f1s, recalls = [], []
        for lab in labels:
            tp = sum(1 for a, b in zip(y, p) if a == lab and b == lab)
            fp = sum(1 for a, b in zip(y, p) if a != lab and b == lab)
            fn = sum(1 for a, b in zip(y, p) if a == lab and b != lab)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            row.update({f"{lab}_precision": pr, f"{lab}_recall": rc, f"{lab}_f1": f1, f"{lab}_support": tp + fn})
            f1s.append(f1)
            recalls.append(rc)
        row["macro_f1"] = float(np.mean(f1s))
        row["balanced_accuracy"] = float(np.mean(recalls))
        row["json_validity_rate"] = float(sub.json_valid.mean())
        row["invalid_outputs"] = int((sub.pred_label == "INVALID").sum())
        per_rep = sub.groupby("repeat").verdict_correct.mean()
        row["accuracy_sd_repeats"] = float(per_rep.std(ddof=0)) if len(per_rep) > 1 else 0.0
        row["accuracy_min_repeat"], row["accuracy_max_repeat"] = float(per_rep.min()), float(per_rep.max())
        rows.append(row)
    return pd.DataFrame(rows), cms


def paired_tests(df: pd.DataFrame, systems: List[str], metrics: List[str]) -> pd.DataFrame:
    from scipy import stats
    pc = per_claim_table(df).set_index(["system", "qid"])
    rows = []
    for m in metrics:
        pvals = {}
        tmp = []
        base = pc.loc[BASELINE][m] if BASELINE in pc.index.get_level_values(0) else None
        for s in systems:
            if s == BASELINE or base is None:
                continue
            other = pc.loc[s][m]
            common = base.index.intersection(other.index)
            d = (other.loc[common] - base.loc[common]).to_numpy(dtype=float)
            d = d[~np.isnan(d)]
            lo, hi = paired_bootstrap(d)
            if len(d) and np.any(d != 0):
                try:
                    p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
                except Exception:
                    p = float("nan")
            else:
                p = 1.0 if len(d) else float("nan")
            pvals[s] = p
            tmp.append({"metric": m, "system": s, "vs": BASELINE, "n_pairs": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else float("nan"),
                        "diff_ci_lo": lo, "diff_ci_hi": hi, "wilcoxon_p": p, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())})
        adj = holm(pvals)
        for r in tmp:
            r["wilcoxon_p_holm"] = adj.get(r["system"], float("nan"))
        rows.extend(tmp)
    return pd.DataFrame(rows)


def mcnemar_tests(df: pd.DataFrame, systems: List[str]) -> pd.DataFrame:
    from scipy import stats
    key = ["qid", "repeat"]
    base = df[df.system == BASELINE].set_index(key).verdict_correct
    rows, pvals = [], {}
    for s in systems:
        if s == BASELINE:
            continue
        other = df[df.system == s].set_index(key).verdict_correct
        common = base.index.intersection(other.index)
        b, o = base.loc[common].to_numpy(), other.loc[common].to_numpy()
        n01 = int(((b == 1) & (o == 0)).sum())   # baseline right, system wrong
        n10 = int(((b == 0) & (o == 1)).sum())   # system right, baseline wrong
        n = n01 + n10
        p = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue) if n else 1.0
        pvals[s] = p
        rows.append({"system": s, "vs": BASELINE, "pairs": int(len(common)), "system_only_correct": n10,
                     "baseline_only_correct": n01, "mcnemar_exact_p": p})
    adj = holm(pvals)
    for r in rows:
        r["mcnemar_p_holm"] = adj.get(r["system"], float("nan"))
    return pd.DataFrame(rows)


def efficiency_table(df: pd.DataFrame, systems: List[str], warmup: int, price_in: float, price_out: float) -> pd.DataFrame:
    rows = []
    for s in systems:
        sub = df[(df.system == s) & (df.query_index >= warmup)]
        lat = sub.t_total_ms.to_numpy(dtype=float)
        row = {"system": s, "n_measurements": int(len(lat)),
               "latency_mean_ms": float(lat.mean()), "latency_median_ms": float(np.median(lat)),
               "latency_sd_ms": float(lat.std(ddof=1)) if len(lat) > 1 else 0.0, "latency_p95_ms": float(np.percentile(lat, 95)),
               "retrieval_ms_mean": float(sub.t_retrieval_ms.mean()), "generation_ms_mean": float(sub.t_generation_ms.mean()),
               "llm_ms_mean": float(sub.llm_ms.mean()),
               "llm_calls_mean": float(sub.llm_calls.mean()), "prompt_tokens_mean": float(sub.prompt_tokens.mean()),
               "output_tokens_mean": float(sub.output_tokens.mean()),
               "thinking_tokens_mean": float(sub.thinking_tokens.mean()) if "thinking_tokens" in sub else 0.0}
        row["est_cost_usd_per_query"] = row["prompt_tokens_mean"] / 1e6 * price_in + (row["output_tokens_mean"] + row["thinking_tokens_mean"]) / 1e6 * price_out
        stage = Counter()
        for js in sub.calls_by_stage:
            stage.update(json.loads(js))
        row["calls_by_stage_mean"] = json.dumps({k: round(v / max(1, len(sub)), 3) for k, v in sorted(stage.items())})
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(df: pd.DataFrame) -> Dict:
    out = {}
    for s in df.system.unique():
        sub = df[df.system == s]
        diags = [json.loads(x) for x in sub.diag]
        d = {}
        if s == "CRAG":
            d["branch_frequency"] = dict(Counter(x.get("crag_branch") for x in diags))
            d["evaluator_parse_failures"] = sum(1 for x in diags if not x.get("eval_parsed", True))
            d["eval_max_mean"] = float(np.mean([x.get("eval_max", 0) for x in diags]))
        if s == "Agentic":
            d["iterations_distribution"] = dict(Counter(x.get("agent_iterations") for x in diags))
            d["iterations_mean"] = float(np.mean([x.get("agent_iterations", 0) for x in diags]))
            d["searches_mean"] = float(np.mean([x.get("agent_searches", 0) for x in diags]))
            d["finished_by"] = dict(Counter(x.get("agent_finished_by") for x in diags))
        if s in ("GraphRAG", "LightRAG"):
            d["fallback_rate"] = float(np.mean([1.0 if x.get("fallback") else 0.0 for x in diags]))
            d["candidates_mean"] = float(np.mean([x.get("n_candidates", 0) for x in diags]))
        if d:
            out[s] = d
    return out


def _fmt(v, ci=None, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f"{v:.{nd}f}"
    if ci is not None and not math.isnan(ci[0]):
        s += f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
    return s


def write_markdown(run_dir: str, mt: pd.DataFrame, vt: pd.DataFrame, cms: Dict, pt: pd.DataFrame, mc: pd.DataFrame,
                   et: pd.DataFrame, diag: Dict, info: Dict, cfg: Dict, labels: List[str]) -> None:
    L = []
    L.append(f"# Benchmark summary — experiment `{cfg.get('experiment')}`\n")
    L.append(f"Claims: {info.get('n_claims')}, documents: {info.get('n_docs')}, chunks: {info.get('n_chunks')}, "
             f"label distribution: {info.get('label_distribution')}, repeats: {cfg.get('repeats')}, k = {cfg.get('k')}, "
             f"generation model: `{cfg.get('gemini_model') if cfg.get('llm') == 'gemini' else 'mock'}`, embedding: `{cfg.get('embedding_model')}`.\n")
    L.append("## Retrieval and evidence metrics (mean over claims, averaged over repeats; 95% bootstrap CI)\n")
    cols = ["doc_ndcg@k", "doc_mrr", "doc_precision@k", "doc_recall@k", "sent_ndcg@k", "sent_mrr", "sent_precision@k", "sent_recall@k",
            "evidence_precision", "evidence_recall", "evidence_f1", "full_rationale_recovered"]
    L.append("| System | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for _, r in mt.iterrows():
        L.append(f"| {r.system} | " + " | ".join(_fmt(r[c], (r[f'{c}_ci_lo'], r[f'{c}_ci_hi'])) for c in cols) + " |")
    L.append("\n## Verdict quality (pooled over repeats)\n")
    vc = ["accuracy", "balanced_accuracy", "macro_f1"] + [f"{l}_f1" for l in labels] + ["json_validity_rate", "accuracy_sd_repeats"]
    L.append("| System | " + " | ".join(vc) + " |")
    L.append("|---|" + "---|" * len(vc))
    for _, r in vt.iterrows():
        L.append(f"| {r.system} | " + " | ".join(_fmt(r[c]) for c in vc) + " |")
    L.append("\n### Confusion matrices (rows = gold, columns = predicted)\n")
    for s, cm in cms.items():
        L.append(f"**{s}**\n")
        L.append(cm.to_markdown())
        L.append("")
    L.append("\n## Paired comparison vs Dense baseline (per-claim differences; Wilcoxon signed-rank, Holm-adjusted)\n")
    L.append(pt.round(4).to_markdown(index=False))
    L.append("\n## Verdict accuracy vs Dense baseline (exact McNemar, Holm-adjusted)\n")
    L.append(mc.round(4).to_markdown(index=False))
    L.append("\n## Efficiency (warm-up queries excluded; sequential execution)\n")
    L.append(et.round(3).to_markdown(index=False))
    L.append("\n## Pipeline diagnostics\n")
    L.append("```json\n" + json.dumps(diag, indent=2) + "\n```")
    with open(os.path.join(run_dir, "analysis", "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def make_figures(run_dir: str, mt: pd.DataFrame, df: pd.DataFrame, warmup: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(run_dir, "analysis")
    for m, title in [("sent_ndcg@k", "Sentence-level nDCG@k"), ("doc_ndcg@k", "Document-level nDCG@k"),
                     ("evidence_recall", "Evidence recall"), ("verdict_correct", "Verdict accuracy")]:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        y = mt[m].to_numpy(dtype=float)
        err = np.vstack([y - mt[f"{m}_ci_lo"].to_numpy(), mt[f"{m}_ci_hi"].to_numpy() - y])
        ax.bar(mt.system, y, yerr=err, capsize=3, color="#4c72b0")
        ax.set_title(f"{title} (95% bootstrap CI)")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_{m.replace('@', '_at_')}.png"), dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sub = df[df.query_index >= warmup]
    systems = list(mt.system)
    ax.boxplot([sub[sub.system == s].t_total_ms.to_numpy() for s in systems], labels=systems, showfliers=False)
    ax.set_ylabel("ms per query")
    ax.set_title("End-to-end latency per query")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_latency.png"), dpi=200)
    plt.close(fig)


def analyze(run_dir: str, price_in: float = 0.0, price_out: float = 0.0) -> None:
    os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
    cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(run_dir, "benchmark_info.json"), encoding="utf-8"))
    labels = ["SUPPORTS", "REFUTES"] if cfg["experiment"] == "main" else ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    df = load_runs(run_dir)
    systems = [s for s in cfg["systems"] if s in set(df.system)]
    mt = main_table(df, systems)
    vt, cms = verdict_table(df, systems, labels)
    pt = paired_tests(df, systems, ["doc_ndcg@k", "doc_mrr", "sent_ndcg@k", "sent_mrr", "evidence_precision", "evidence_recall", "evidence_f1"])
    mc = mcnemar_tests(df, systems)
    et = efficiency_table(df, systems, cfg["warmup_queries"], price_in, price_out)
    diag = diagnostics(df)
    out = os.path.join(run_dir, "analysis")
    mt.to_csv(os.path.join(out, "table_main_metrics.csv"), index=False)
    vt.to_csv(os.path.join(out, "table_verdict.csv"), index=False)
    for s, cm in cms.items():
        cm.to_csv(os.path.join(out, f"confusion_{s.replace('+', 'plus')}.csv"))
    pt.to_csv(os.path.join(out, "table_paired_tests.csv"), index=False)
    mc.to_csv(os.path.join(out, "table_mcnemar.csv"), index=False)
    et.to_csv(os.path.join(out, "table_efficiency.csv"), index=False)
    per_claim_table(df).to_csv(os.path.join(out, "per_claim_metrics.csv"), index=False)
    with open(os.path.join(out, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    write_markdown(run_dir, mt, vt, cms, pt, mc, et, diag, info, cfg, labels)
    make_figures(run_dir, mt, df, cfg["warmup_queries"])
    print(f"Analysis written to {out}")
