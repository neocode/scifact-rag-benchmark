"""Retrieval primitives shared by all pipelines.

* ``DenseIndex``  - exact cosine similarity over normalised embeddings (no ANN approximation).
* ``BM25Index``   - Okapi BM25 over the same sentence chunks (rank_bm25).
* ``rrf``         - reciprocal-rank fusion of several ranked lists.
* ``Reranker``    - cross-encoder reranking (BAAI/bge-reranker-base) with an offline mock.

A ranked list is a ``List[Tuple[chunk_index, score]]`` sorted by decreasing score.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

Ranked = List[Tuple[int, float]]


class DenseIndex:
    def __init__(self, embedder, passage_emb: np.ndarray):
        self.embedder = embedder
        self.emb = passage_emb.astype(np.float32)

    def search(self, query: str, k: int) -> Ranked:
        q = self.embedder.encode_queries([query])[0]
        sims = self.emb @ q
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        return [(int(i), float(sims[i])) for i in idx]

    def score_subset(self, query: str, cand: Sequence[int]) -> Ranked:
        """Rank an explicit candidate subset by cosine similarity (used by graph pipelines)."""
        if not cand:
            return []
        q = self.embedder.encode_queries([query])[0]
        cand = list(dict.fromkeys(int(c) for c in cand))
        sims = self.emb[cand] @ q
        order = np.argsort(-sims, kind="stable")
        return [(cand[i], float(sims[i])) for i in order]

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode_passages(texts)


_TOK = re.compile(r"\w+")


class BM25Index:
    def __init__(self, texts: List[str]):
        from rank_bm25 import BM25Okapi
        self.tokens = [_TOK.findall(t.lower()) for t in texts]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int) -> Ranked:
        scores = self.bm25.get_scores(_TOK.findall(query.lower()))
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(int(i), float(scores[i])) for i in idx]


def rrf(lists: Sequence[Ranked], k_const: int = 60) -> Ranked:
    """Reciprocal-rank fusion: score(d) = sum_i 1 / (k + rank_i(d))."""
    acc: Dict[int, float] = {}
    for lst in lists:
        for rank, (i, _) in enumerate(lst, start=1):
            acc[i] = acc.get(i, 0.0) + 1.0 / (k_const + rank)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)
        self.name = model_name

    def rerank(self, query: str, cand: Ranked, texts: List[str], k: int) -> Ranked:
        pairs = [(query, texts[i]) for i, _ in cand]
        scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(cand[j][0], float(scores[j])) for j in order]


class MockReranker:
    name = "mock-reranker"

    def rerank(self, query: str, cand: Ranked, texts: List[str], k: int) -> Ranked:
        q = set(_TOK.findall(query.lower()))
        scored = [(i, len(q & set(_TOK.findall(texts[i].lower()))) + s * 1e-3) for i, s in cand]
        return sorted(scored, key=lambda x: -x[1])[:k]


def make_reranker(cfg):
    return MockReranker() if cfg.reranker == "mock" else Reranker(cfg.reranker_model)


def doc_ranking(ranked: Ranked, doc_ids: List[str], k: int) -> List[str]:
    """Aggregate a chunk ranking into a document ranking (max score per document, first occurrence order)."""
    seen: List[str] = []
    for i, _ in ranked:
        d = doc_ids[i]
        if d not in seen:
            seen.append(d)
        if len(seen) >= k:
            break
    return seen
