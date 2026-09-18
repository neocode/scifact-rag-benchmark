#!/usr/bin/env python
"""Reproducible benchmark of next-generation RAG pipelines on SciFact.

Examples
--------
Smoke test without any downloads or API access (synthetic data is generated when SciFact is absent):
    python run_benchmark.py --experiment main --llm mock --embedder mock --reranker mock --limit-claims 8 --repeats 2 --out results_smoke

Main experiment (fixed train-split subset, 120 claims, 3 repeats):
    export GEMINI_API_KEY=...            # Windows PowerShell:  $env:GEMINI_API_KEY="..."
    python run_benchmark.py --experiment main --repeats 3

Standard open dev split (300 claims incl. NOT ENOUGH INFO, full 5 183-abstract corpus), one repeat:
    python run_benchmark.py --experiment dev --repeats 1

Re-run only the analysis:
    python run_benchmark.py --experiment main --analyze-only
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scifact_bench.config import Config                                  # noqa: E402
from scifact_bench import data as D                                      # noqa: E402
from scifact_bench.embed import make_embedder, cached_passage_embeddings  # noqa: E402
from scifact_bench.retrieval import DenseIndex, BM25Index, make_reranker, doc_ranking  # noqa: E402
from scifact_bench.llm import make_llm                                   # noqa: E402
from scifact_bench.graph import KnowledgeGraph                           # noqa: E402
from scifact_bench.systems import SYSTEMS, Resources, Generator          # noqa: E402
from scifact_bench.metrics import query_metrics                          # noqa: E402
from scifact_bench.prompts import export_prompts                         # noqa: E402
from scifact_bench.analyze import analyze                                # noqa: E402


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["main", "dev"], default="main")
    p.add_argument("--systems", default=None, help="comma-separated subset, e.g. BM25,Dense,Hybrid")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--n-claims", type=int, default=None, help="main experiment: number of sampled claims")
    p.add_argument("--n-docs", type=int, default=None, help="main experiment: corpus size")
    p.add_argument("--limit-claims", type=int, default=0, help="debug: use only the first N benchmark claims")
    p.add_argument("--llm", choices=["gemini", "mock"], default=None)
    p.add_argument("--gemini-model", default=None)
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Gemini thinking budget in tokens; 0 (default) disables thinking, -1 = model default (dynamic)")
    p.add_argument("--embedder", choices=["e5", "mock"], default=None)
    p.add_argument("--reranker", choices=["cross-encoder", "mock"], default=None)
    p.add_argument("--agent-max-iterations", type=int, default=None)
    p.add_argument("--graph-summaries", action="store_true", help="also generate GraphRAG community summaries (extra cost)")
    p.add_argument("--request-pause", type=float, default=None, help="seconds to sleep after each LLM call")
    p.add_argument("--out", default=None, help="results directory (default: results)")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens (for cost estimate)")
    p.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens (for cost estimate)")
    a = p.parse_args()
    cfg = Config(experiment=a.experiment)
    if a.systems:
        cfg.systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    for name in ["repeats", "n_claims", "n_docs", "llm", "gemini_model", "embedder", "reranker", "agent_max_iterations", "thinking_budget"]:
        v = getattr(a, name)
        if v is not None:
            setattr(cfg, name, v)
    if a.request_pause is not None:
        cfg.request_pause_s = a.request_pause
    if a.out:
        cfg.out_dir = a.out
    cfg.graph_summaries = bool(a.graph_summaries)
    cfg._limit = a.limit_claims          # type: ignore[attr-defined]
    cfg._analyze_only = a.analyze_only   # type: ignore[attr-defined]
    cfg._prices = (a.price_in, a.price_out)  # type: ignore[attr-defined]
    return cfg


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for mod in ["numpy", "pandas", "scipy", "sentence_transformers", "transformers", "torch", "rank_bm25", "networkx", "google.genai", "leidenalg", "igraph"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = "not installed"
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def synthetic_scifact(data_dir: str, seed: int = 0) -> None:
    """Create a tiny synthetic SciFact-like release for offline smoke tests (only if real data is absent)."""
    import random
    rng = random.Random(seed)
    base = os.path.join(data_dir, "extracted", "data")
    os.makedirs(base, exist_ok=True)
    topics = ["interleukin-6 signalling in hepatocytes", "microRNA regulation of tumour suppressor genes",
              "statin therapy and cardiovascular mortality", "gut microbiome diversity in obesity",
              "insulin resistance in skeletal muscle", "vitamin D supplementation and bone density",
              "PD-1 blockade in melanoma", "mitochondrial dysfunction in Parkinson disease"]
    with open(os.path.join(base, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for i in range(1, 121):
            t = topics[i % len(topics)]
            sents = [f"We investigated {t} in a cohort of {rng.randint(20, 900)} participants.",
                     f"The {t.split()[0]} exposure increased the outcome measure by {rng.randint(5, 80)} percent.",
                     f"No significant association between {t} and mortality was observed in the control group.",
                     f"These results suggest that {t} contributes to disease progression."]
            f.write(json.dumps({"doc_id": 1000 + i, "title": f"Study {i} on {t}", "abstract": sents, "structured": False}) + "\n")
    for split, n0, n in [("train", 1, 60), ("dev", 61, 30)]:
        with open(os.path.join(base, f"claims_{split}.jsonl"), "w", encoding="utf-8") as f:
            for j in range(n0, n0 + n):
                did = 1000 + j
                t = topics[j % len(topics)]
                r = rng.random()
                if r < 0.45:
                    rec = {"id": j, "claim": f"{t.capitalize()} increases the outcome measure.", "evidence": {str(did): [{"sentences": [1], "label": "SUPPORT"}]}, "cited_doc_ids": [did]}
                elif r < 0.85:
                    rec = {"id": j, "claim": f"{t.capitalize()} is associated with mortality.", "evidence": {str(did): [{"sentences": [2], "label": "CONTRADICT"}]}, "cited_doc_ids": [did]}
                else:
                    rec = {"id": j, "claim": f"{t.capitalize()} reverses disease progression.", "evidence": {}, "cited_doc_ids": [did]}
                f.write(json.dumps(rec) + "\n")


def main() -> None:
    cfg = parse_args()
    run_dir = cfg.run_dir
    os.makedirs(run_dir, exist_ok=True)
    if cfg._analyze_only:  # type: ignore[attr-defined]
        analyze(run_dir, *cfg._prices)  # type: ignore[attr-defined]
        return

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(cfg.to_json())
    with open(os.path.join(run_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(environment_info(), f, indent=2)
    export_prompts(os.path.join(run_dir, "prompts.md"))

    # ------------------------------------------------------------------ data
    if cfg.llm == "mock" and cfg.embedder == "mock" and not any(True for _ in D.Path(cfg.data_dir).rglob("corpus.jsonl")):
        print("Offline smoke test: generating a synthetic SciFact-like release")
        synthetic_scifact(cfg.data_dir)
    base = D.ensure_download(cfg.data_dir)
    docs = D.load_corpus(base)
    if cfg.experiment == "main":
        claims_all = D.load_claims(base, "train", docs)
        claims, corpus, info = D.build_main_benchmark(docs, claims_all, cfg.n_claims, cfg.n_docs, cfg.hard_negative_pool, cfg.seed)
        dev_claims = D.load_claims(base, "dev", docs)
        info["overlap_with_dev_split"] = D.overlap_report(claims, dev_claims)
        info["note"] = "All hyper-parameters were fixed a priori; no parameter was tuned on the benchmark claims."
    else:
        claims_all = D.load_claims(base, "dev", docs)
        claims, corpus, info = D.build_dev_benchmark(docs, claims_all)
    if cfg._limit:  # type: ignore[attr-defined]
        claims = claims[: cfg._limit]  # type: ignore[attr-defined]
        info["debug_limit_claims"] = cfg._limit  # type: ignore[attr-defined]
        info["n_claims"] = len(claims)
        from collections import Counter
        info["label_distribution"] = dict(Counter(c.label for c in claims))
    chunks = D.make_chunks(corpus)
    D.export_benchmark(run_dir, claims, corpus, chunks, info)
    print(f"[{cfg.experiment}] claims={len(claims)} docs={len(corpus)} chunks={len(chunks)} labels={info['label_distribution']}")

    # ------------------------------------------------------------------ indices and models
    embedder = make_embedder(cfg)
    texts = [c.text for c in chunks]
    emb = cached_passage_embeddings(embedder, texts, cfg.cache_dir, f"{cfg.experiment}_{cfg.embedder}")
    dense = DenseIndex(embedder, emb)
    bm25 = BM25Index(texts)
    reranker = make_reranker(cfg) if "Dense+Rerank" in cfg.systems else None
    llm = make_llm(cfg, cache_path=os.path.join(cfg.cache_dir, f"llm_cache_{cfg.experiment}.jsonl"))
    chunk_dicts = [c.as_dict() for c in chunks]
    doc_ids = [c.doc_id for c in chunks]

    kg = None
    if any(s in cfg.systems for s in ("GraphRAG", "LightRAG")):
        print("Building the entity-relation graph (one extraction call per abstract; cached) ...")
        kg = KnowledgeGraph()
        kg.build(llm, corpus, {c.chunk_id: i for i, c in enumerate(chunks)}, {d.doc_id: d.sentences for d in corpus.values()},
                 dense, resolution=cfg.graph_leiden_resolution, summaries=cfg.graph_summaries)
        kg.save_stats(os.path.join(run_dir, "graph_stats.json"))
        print(f"  graph: {kg.stats['n_entities']} entities, {kg.stats['n_edges']} edges, {kg.stats['n_communities']} communities ({kg.community_method})")

    res = Resources(cfg=cfg, chunks=chunk_dicts, dense=dense, bm25=bm25, llm=llm, reranker=reranker, kg=kg)
    systems = {name: SYSTEMS[name](res) for name in cfg.systems}
    generator = Generator(llm, cfg.labels)

    # ------------------------------------------------------------------ resume support
    runs_path = os.path.join(run_dir, "runs.jsonl")
    done = set()
    if os.path.exists(runs_path):
        with open(runs_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["system"], r["repeat"], r["qid"]))
                except Exception:
                    continue
        if done:
            print(f"Resuming: {len(done)} (system, repeat, claim) records already present")

    # ------------------------------------------------------------------ main loop (systems interleaved per claim)
    total = cfg.repeats * len(claims) * len(systems)
    n_done = len(done)
    t_start = time.time()
    with open(runs_path, "a", encoding="utf-8") as out:
        for rep in range(cfg.repeats):
            for qi, claim in enumerate(claims):
                for name, system in systems.items():
                    if (name, rep, claim.qid) in done:
                        continue
                    llm.reset_log()
                    t0 = time.perf_counter()
                    ranking, diag = system.retrieve(claim.text)
                    t1 = time.perf_counter()
                    context = [chunk_dicts[i] for i, _ in ranking[: cfg.k]]
                    gen = generator.generate(claim.text, context)
                    t2 = time.perf_counter()
                    sent_rank = [chunk_dicts[i]["chunk_id"] for i, _ in ranking]
                    doc_rank = doc_ranking(ranking, doc_ids, cfg.k)
                    m = query_metrics(doc_rank, sent_rank, claim, gen, cfg.k)
                    rec = {
                        "system": name, "repeat": rep, "qid": claim.qid, "query_index": qi, "claim": claim.text,
                        "gold_label": claim.label, "pred_label": gen.label, "cited": gen.cited,
                        "context_ids": [c["chunk_id"] for c in context], "doc_ranking": doc_rank,
                        "timing": {"retrieval_ms": round((t1 - t0) * 1000, 1), "generation_ms": round((t2 - t1) * 1000, 1),
                                   "total_ms": round((t2 - t0) * 1000, 1)},
                        "llm": llm.log.as_dict(), "diag": diag, "metrics": m, "answer": gen.answer,
                        "raw_output": gen.raw[:500] if not gen.json_valid else "",
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    n_done += 1
                    if n_done % 25 == 0 or n_done == total:
                        el = time.time() - t_start
                        print(f"  {n_done}/{total} records  ({el/60:.1f} min elapsed)")
    print("Run complete. Analysing ...")
    analyze(run_dir, *cfg._prices)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
