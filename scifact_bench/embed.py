"""Embedding backends: E5-base-v2 (sentence-transformers) and an offline TF-IDF mock."""
from __future__ import annotations

import hashlib
import os
from typing import List, Optional

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class BaseEmbedder:
    name = "base"
    dim = 0

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class E5Embedder(BaseEmbedder):
    """intfloat/e5-base-v2 with the instruction prefixes recommended by the authors."""

    def __init__(self, model_name: str = "intfloat/e5-base-v2", batch_size: int = 64, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size

    def _enc(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=len(texts) > 256)
        return v.astype(np.float32)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        return self._enc([f"passage: {t}" for t in texts])

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._enc([f"query: {t}" for t in texts])


class MockEmbedder(BaseEmbedder):
    """Hashing TF-IDF stand-in (no downloads) used only for offline pipeline tests."""
    name = "mock-tfidf"

    def __init__(self, dim: int = 4096):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm=None, stop_words="english")
        self.dim = dim

    def _enc(self, texts: List[str]) -> np.ndarray:
        m = self.vec.transform(texts).astype(np.float32)
        m.data = np.log1p(m.data)
        return _normalize(m.toarray())

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        return self._enc(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._enc(texts)


def make_embedder(cfg) -> BaseEmbedder:
    if cfg.embedder == "mock":
        return MockEmbedder()
    return E5Embedder(cfg.embedding_model)


def cached_passage_embeddings(embedder: BaseEmbedder, texts: List[str], cache_dir: str, tag: str) -> np.ndarray:
    """Encode passages once per (model, corpus) and cache to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha256((embedder.name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(cache_dir, f"emb_{tag}_{h}.npy")
    if os.path.exists(path):
        return np.load(path)
    emb = embedder.encode_passages(texts)
    np.save(path, emb)
    return emb
