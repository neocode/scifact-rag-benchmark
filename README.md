# Evidence-grounded benchmark of next-generation RAG pipelines on SciFact

Reproducible code, benchmark identifiers, prompts and results for the article
*"Benchmarking next-generation retrieval-augmented generation architectures for scientific claim
verification on abstract corpora"* (Sheremet O., Sadovoi O., Podlesny S., Sokhina Yu., Sheremet K.).

The benchmark compares eight pipelines that share one embedding model (`intfloat/e5-base-v2`),
one sentence-level index, one generation model and one constrained JSON output contract, so that
differences arise from retrieval / orchestration logic only:

| Pipeline | Retrieval / orchestration (all restricted to the indexed corpus) | LLM calls per query |
|---|---|---|
| **BM25** | Okapi BM25 over sentence chunks (sparse baseline) | 1 (generation) |
| **Dense** | exact cosine search over E5 embeddings (dense baseline = "Baseline RAG") | 1 |
| **Hybrid** | reciprocal-rank fusion of BM25 and dense rankings | 1 |
| **Dense+Rerank** | dense top-20 re-ordered by the cross-encoder `BAAI/bge-reranker-base` | 1 |
| **CRAG** (corpus-restricted) | LLM retrieval evaluator scores the top-k; *Correct* → generate; *Ambiguous* → LLM query reformulation + fused re-retrieval; *Incorrect* → reformulation + hybrid re-retrieval (substitute for the disabled web search); knowledge refinement demotes chunks scored < 0.3 | 2–3 |
| **Agentic** | ReAct-style controller: initial dense search, then up to *I*<sub>max</sub> = 3 controller decisions (`search` with a new query / `finish`); rankings fused by RRF | 2–4 |
| **GraphRAG** (local search) | offline LLM entity–relation extraction per abstract → entity graph → Leiden communities; online: query entities (name match + embedding match) → 1-hop neighbourhood → provenance sentences → dense re-ranking | 1 (+ offline extraction) |
| **LightRAG** (dual-level) | same graph; LLM extracts low-level (entity) and high-level (theme/relation) keywords; entity- and relation-matched sentences merged and re-ranked | 2 (+ offline extraction) |

The global (community-summary, map–reduce) search path of GraphRAG is **not exercised** by
claim-level queries and is disabled by default (`--graph-summaries` enables summary generation only).

## Experiments

* **main** – fixed subset of the SciFact *train* split: 120 SUPPORT/CONTRADICT claims with annotated
  evidence sampled with seed 42; corpus = all their evidence abstracts + keyword-overlap hard negatives
  up to 500 abstracts. Two-class verdict. Three independent end-to-end repeats.
* **dev** – the standard open SciFact *dev* split (300 claims including NOT ENOUGH INFO) over the
  **full** 5 183-abstract corpus. Three-class verdict. Retrieval metrics are computed on claims with
  annotated evidence; verdict metrics on all claims.

Exact claim ids, document ids, chunk ids, label distribution and the (empty) overlap between the
main subset and the dev split are written to `results/<experiment>/benchmark_*.{jsonl,txt,json}`.
All hyper-parameters are fixed a priori in `scifact_bench/config.py`; nothing is tuned on the
evaluation claims.

## Headline results

Verdict accuracy (pooled over repeats; full tables with 95 % bootstrap CIs, per-class scores and
paired tests are in `results/<experiment>/analysis/summary.md`):

| System | main (120 claims, 2-class) | dev (300 claims, 3-class, full corpus) |
|---|---|---|
| BM25 | 0.758 | 0.670 |
| Dense | 0.858 | 0.757 |
| Hybrid | 0.858 | 0.713 |
| Dense+Rerank | **0.892** | 0.733 |
| CRAG | 0.886 | 0.763 |
| Agentic | 0.875 | **0.767** |
| GraphRAG | 0.833 | 0.740 |
| LightRAG | 0.850 | 0.750 |

On the two-class main subset the cross-encoder reranking pipeline is strongest (0.892), closely
followed by CRAG (0.886); on the three-class dev split over the full corpus the multi-step
pipelines lead (Agentic 0.767, CRAG 0.763). Evidence F1 of the cited rationales is highest for
Agentic (0.635 main / 0.536 dev) and CRAG (0.623 / 0.532). JSON validity is ≥ 0.997 for all
systems on both splits. The exact McNemar test with Holm correction shows no verdict-accuracy
difference versus the dense baseline for any pipeline except BM25, which is significantly worse
(main p < 0.001, dev p = 0.018). The trade-off is therefore evidence quality and latency rather
than accuracy: CRAG and Agentic cost ≈ 2.6 and ≈ 2.9 LLM calls per query (median end-to-end
latency ≈ 2.4 s) versus a single call (≈ 0.9 s) for the one-pass pipelines.

## Metrics (`results/<experiment>/analysis/`)

* Document- and sentence-level Precision@5, Recall@5, nDCG@5, MRR on the ranking passed to the generator.
* Evidence precision / recall / F1 of the *cited* sentence ids against the gold rationale union;
  full-rationale recovery; FEVER-style score (verdict correct ∧ a complete rationale set cited).
* Verdict accuracy, per-class precision / recall / F1, macro-F1, balanced accuracy, confusion matrices,
  JSON validity rate (invalid outputs are counted as wrong predictions).
* 95 % percentile-bootstrap CIs over claims; paired comparison against the dense baseline
  (paired bootstrap CI of the difference, Wilcoxon signed-rank, exact McNemar for verdicts, Holm correction).
* Generation stability: SD of system-level means across repeats.
* Efficiency: end-to-end latency mean / median / SD / p95 (warm-up excluded, sequential execution),
  retrieval vs generation time, LLM calls per query by stage, prompt / output tokens, optional cost estimate.
* Diagnostics: CRAG branch frequencies, agent iteration distribution, graph statistics, graph fallback rate.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# 1. offline smoke test (synthetic data, no downloads, no API) – takes ~10 s
python run_benchmark.py --experiment main --llm mock --embedder mock --reranker mock --limit-claims 8 --repeats 2 --out results_smoke

# 2. main experiment (downloads SciFact ~10 MB, E5 ~440 MB, reranker ~1.1 GB on first run)
export GEMINI_API_KEY="..."                  # PowerShell: $env:GEMINI_API_KEY="..."
python run_benchmark.py --experiment main --repeats 3 --gemini-model gemini-2.5-flash

# 3. standard dev split over the full corpus (graph extraction for 5 183 abstracts is cached)
python run_benchmark.py --experiment dev --repeats 1 --gemini-model gemini-2.5-flash

# optional: cost estimate in the efficiency table (USD per 1M tokens)
python run_benchmark.py --experiment main --analyze-only --price-in 0.30 --price-out 2.50
```

Gemini 2.5 models *think* by default, which multiplies latency and output-token cost several-fold and adds
an uncontrolled, non-deterministic component to every call. The benchmark therefore sets
`thinking_budget = 0` (thinking disabled) for all calls by default, so that the generator is identical across
pipelines and latency reflects retrieval/orchestration differences only. Use `--thinking-budget -1` to restore
the model default (dynamic thinking) or a positive number for a fixed budget; thinking tokens are logged
per call in either case.

Runs are resumable: every (system, repeat, claim) record is appended to `runs.jsonl`, and a restarted
run skips the records already present. Offline extraction calls are cached in `cache/`.

Approximate budget (Gemini Flash-class model, sequential execution): main ≈ 4 500 LLM calls
(≈ 2 h), dev ≈ 3 600 online calls + 5 183 extraction calls (≈ 3–4 h). Use `--systems` to run a subset,
e.g. `--systems BM25,Dense,Hybrid,Dense+Rerank,CRAG,Agentic` to skip graph construction.

## Repository layout

```
run_benchmark.py            command-line entry point (prepare → index → run → analyse)
scifact_bench/              package (see module docstrings)
results/main/               reproducibility bundle of the main experiment (see below)
results/dev/                same for the dev-split experiment
main_run.log                console log of the reported main run (2026-09-08)
dev_run.log                 console log of the reported dev run (2026-09-09)
```

Each `results/<experiment>/` directory is a self-contained reproducibility bundle:

```
config.json                 every hyper-parameter used for the run (model, seed, k, thresholds, …)
environment.json            Python / OS / library / GPU snapshot captured at run start
benchmark_claims.jsonl      the exact claim subset evaluated (ids, labels, gold rationales)
benchmark_doc_ids.txt       document ids of the indexed corpus
benchmark_chunk_ids.txt     sentence-chunk ids of the index
benchmark_info.json         subset construction: split, seed, sizes, label distribution,
                            overlap check against the other split, chunk count
prompts.md                  verbatim prompt templates used for every LLM call
graph_stats.json            entity–relation graph statistics (nodes, edges, communities, fallback rate)
runs.jsonl                  raw records: one JSON line per (system, repeat, claim) with the ranking,
                            cited evidence, verdict, token counts, latency and model version
analysis/                   all tables and figures reported in the article:
                            summary.md, table_main_metrics.csv, table_verdict.csv,
                            table_paired_tests.csv, table_mcnemar.csv, table_efficiency.csv,
                            per_claim_metrics.csv, diagnostics.json, confusion_<system>.csv,
                            fig_*.png
```

Downloaded data (`data/`), the LLM extraction cache (`cache/`) and smoke-test output
(`results_smoke/`) are git-ignored: `data/` is fetched automatically on first run and the cache
is rebuilt on demand. The reported runs were executed on 2026-09-08/09 with
`gemini-2.5-flash`, `temperature = 0`, `thinking_budget = 0`; the full console output is preserved
in `main_run.log` / `dev_run.log`.

## Citation

Please cite the article (DOI to be added on publication) and the SciFact dataset
(Wadden et al., EMNLP 2020).

## License

MIT (code). SciFact data are distributed under their original license by the Allen Institute for AI.
