"""Reproducible benchmark of next-generation RAG pipelines on SciFact.

Package layout
--------------
config.py     experiment configuration (all hyper-parameters are fixed a priori)
data.py       SciFact download, benchmark subset construction, sentence chunking, ID export
llm.py        Gemini client with retries / token accounting, plus an offline mock
embed.py      E5-base-v2 embedder (sentence-transformers), plus an offline mock
retrieval.py  exact dense search, BM25, reciprocal-rank fusion, cross-encoder reranking
graph.py      LLM entity/relation extraction, graph construction, Leiden communities,
              GraphRAG-style local search and LightRAG-style dual-level retrieval
systems.py    the compared pipelines (baselines + Agentic / GraphRAG / LightRAG / CRAG)
prompts.py    every prompt used by the benchmark (exported verbatim to prompts.md)
metrics.py    per-query retrieval, evidence and verdict metrics
analyze.py    aggregation, bootstrap CIs, paired tests, tables and figures
"""

__version__ = "2.0.0"
