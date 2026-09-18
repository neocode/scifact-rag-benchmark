"""Experiment configuration.

All hyper-parameters are fixed a priori (no tuning on the evaluation claims).
Two experiments are defined:

* ``main`` - fixed SciFact *train-split* subset (SUPPORT / CONTRADICT claims with
  annotated evidence; 500-abstract corpus with keyword-overlap hard negatives).
* ``dev``  - the standard open SciFact *dev* split (all 300 claims, including
  NOT ENOUGH INFO) over the *full* 5 183-abstract corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class Config:
    # ---------------------------------------------------------------- general
    experiment: str = "main"            # "main" | "dev"
    seed: int = 42
    out_dir: str = "results"
    data_dir: str = "data/scifact"
    cache_dir: str = "cache"

    # ------------------------------------------------------------ benchmark
    n_claims: int = 120                 # main experiment: number of sampled claims
    n_docs: int = 500                   # main experiment: corpus size (evidence + hard negatives)
    hard_negative_pool: int = 2000      # top-N keyword-overlap distractors from which negatives are sampled
    calibration_claims: int = 0         # optional disjoint claims reserved for sanity checks (not used for tuning)

    # ------------------------------------------------------------ retrieval
    k: int = 5                          # final context size and evaluation cut-off
    pool_k: int = 50                    # candidate pool size for doc-level ranking / fusion / reranking
    embedding_model: str = "intfloat/e5-base-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_depth: int = 20              # dense candidates passed to the cross-encoder
    rrf_k: int = 60                     # reciprocal-rank-fusion constant

    # ------------------------------------------------------------ CRAG (corpus-restricted)
    crag_theta_high: float = 0.7        # max evaluator score above which retrieval is "Correct"
    crag_theta_low: float = 0.3         # max evaluator score below which retrieval is "Incorrect"
    crag_theta_mean: float = 0.5        # mean evaluator score required for "Correct"
    crag_keep_threshold: float = 0.3    # knowledge-refinement filter: drop chunks scored below this

    # ------------------------------------------------------------ Agentic RAG
    agent_max_iterations: int = 3       # hard iteration budget I_max
    agent_pool_per_step: int = 20       # dense candidates gathered per search action

    # ------------------------------------------------------------ Graph-based pipelines
    graph_extraction_unit: str = "abstract"   # one extraction call per abstract
    graph_entity_top_m: int = 8         # query entities matched by embedding similarity
    graph_relation_top_m: int = 8       # LightRAG high-level: relations matched by embedding similarity
    graph_min_candidates: int = 5       # fall back to dense retrieval below this candidate count
    graph_leiden_resolution: float = 1.0
    graph_summaries: bool = False       # community summaries (GraphRAG global path) - not exercised by claims

    # ------------------------------------------------------------ generation
    llm: str = "gemini"                 # "gemini" | "mock"
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    thinking_budget: int = 0            # Gemini 2.5 "thinking": 0 = disabled (fixed generator, no hidden reasoning tokens)
    max_retries: int = 8
    repeats: int = 3                    # independent end-to-end runs per system (generation stability)
    warmup_queries: int = 3             # leading queries excluded from latency statistics
    request_pause_s: float = 0.0        # optional pause between LLM calls (rate limiting)

    # ------------------------------------------------------------ systems
    systems: List[str] = field(default_factory=lambda: [
        "BM25", "Dense", "Hybrid", "Dense+Rerank",
        "CRAG", "Agentic", "GraphRAG", "LightRAG",
    ])

    # ------------------------------------------------------------ offline testing
    embedder: str = "e5"                # "e5" | "mock"
    reranker: str = "cross-encoder"     # "cross-encoder" | "mock"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @property
    def labels(self) -> List[str]:
        return ["SUPPORTS", "REFUTES"] if self.experiment == "main" else ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    @property
    def run_dir(self) -> str:
        return f"{self.out_dir}/{self.experiment}"
