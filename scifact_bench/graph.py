"""Graph construction and graph-based retrieval (GraphRAG-style local search, LightRAG-style dual-level retrieval).

Offline stage (cached on disk, counted separately from per-query cost):
  1. one LLM extraction call per abstract -> entities (name, type) and relations (source, target, relation, sentence_ids)
  2. entity resolution by normalised-name matching (lower-case, punctuation/whitespace collapsed)
  3. graph G = (V = entities, E = relations) with document / sentence provenance on nodes and edges
  4. community detection (Leiden via leidenalg+igraph if installed, otherwise Louvain from networkx)
  5. embeddings of entity names and relation descriptions (same E5 model) for query matching
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .llm import BaseLLM, parse_json_lenient
from .prompts import ENTITY_RELATION_EXTRACTION, COMMUNITY_SUMMARY
from .retrieval import DenseIndex, Ranked


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


class KnowledgeGraph:
    def __init__(self):
        import networkx as nx
        self.nx = nx
        self.G = nx.Graph()
        self.entity_chunks: Dict[str, Set[int]] = defaultdict(set)      # entity -> chunk indices
        self.relations: List[Dict] = []                                  # {src,tgt,rel,chunks}
        self.entity_names: List[str] = []
        self.entity_emb = None
        self.relation_texts: List[str] = []
        self.relation_emb = None
        self.communities: Dict[str, int] = {}
        self.community_method = ""
        self.stats: Dict = {}
        self.summaries: Dict[int, str] = {}

    # ----------------------------------------------------------------- build
    def build(self, llm: BaseLLM, corpus_docs: Dict[str, "Doc"], chunk_index: Dict[str, int], sentences_by_doc: Dict[str, List[str]],
              embedder_index: DenseIndex, resolution: float = 1.0, summaries: bool = False, verbose: bool = True) -> None:
        t0 = time.perf_counter()
        llm.reset_log()
        n_fail = 0
        docs = sorted(corpus_docs, key=int)
        for n, did in enumerate(docs, 1):
            d = corpus_docs[did]
            sents = "\n".join(f"[{i}] {s}" for i, s in enumerate(d.sentences))
            prompt = ENTITY_RELATION_EXTRACTION.format(title=d.title, sentences=sents)
            r = llm.call(prompt, stage="graph_extraction", json_mode=True, use_cache=True)
            obj = parse_json_lenient(r.text)
            if not isinstance(obj, dict):
                n_fail += 1
                continue
            self._ingest(did, obj, d.sentences, chunk_index)
            if verbose and n % 100 == 0:
                print(f"  extracted {n}/{len(docs)} abstracts")
        self._finalize(embedder_index, resolution)
        if summaries:
            self._summarize(llm)
        self.stats = {
            "n_docs_extracted": len(docs), "n_extraction_failures": n_fail,
            "n_entities": self.G.number_of_nodes(), "n_relations": len(self.relations),
            "n_edges": self.G.number_of_edges(), "n_communities": len(set(self.communities.values())),
            "community_method": self.community_method, "resolution": resolution,
            "extraction_llm": llm.log.as_dict(), "build_seconds": round(time.perf_counter() - t0, 1),
        }

    def _ingest(self, did: str, obj: Dict, sentences: List[str], chunk_index: Dict[str, int]) -> None:
        low = [s.lower() for s in sentences]

        def chunks_for_sids(sids) -> Set[int]:
            out = set()
            for s in sids or []:
                try:
                    cid = f"{did}::sent_{int(s)}"
                except Exception:
                    continue
                if cid in chunk_index:
                    out.add(chunk_index[cid])
            return out

        def chunks_mentioning(name: str) -> Set[int]:
            out = set()
            nn = norm_name(name)
            if len(nn) < 3:
                return out
            toks = nn.split()
            for i, s in enumerate(low):
                if nn in s or (len(toks) > 1 and all(t in s for t in toks)):
                    cid = f"{did}::sent_{i}"
                    if cid in chunk_index:
                        out.add(chunk_index[cid])
            return out

        for e in obj.get("entities", []) or []:
            if not isinstance(e, dict):
                continue
            name = norm_name(e.get("name", ""))
            if not name:
                continue
            if name not in self.G:
                self.G.add_node(name, type=str(e.get("type", "other")), docs=set())
            self.G.nodes[name]["docs"].add(did)
            self.entity_chunks[name] |= chunks_mentioning(name)
        for r in obj.get("relations", []) or []:
            if not isinstance(r, dict):
                continue
            s, t = norm_name(r.get("source", "")), norm_name(r.get("target", ""))
            if not s or not t or s == t:
                continue
            for x in (s, t):
                if x not in self.G:
                    self.G.add_node(x, type="other", docs=set())
                self.G.nodes[x]["docs"].add(did)
            ch = chunks_for_sids(r.get("sentence_ids")) or (chunks_mentioning(s) & chunks_mentioning(t))
            self.entity_chunks[s] |= ch
            self.entity_chunks[t] |= ch
            rel = str(r.get("relation", "related to")).strip()
            if self.G.has_edge(s, t):
                self.G[s][t]["weight"] += 1.0
                self.G[s][t]["chunks"] |= ch
            else:
                self.G.add_edge(s, t, weight=1.0, chunks=set(ch), relation=rel)
            self.relations.append({"src": s, "tgt": t, "rel": rel, "chunks": ch, "doc": did})

    def _finalize(self, index: DenseIndex, resolution: float) -> None:
        self.entity_names = sorted(self.G.nodes)
        if self.entity_names:
            self.entity_emb = index.encode(self.entity_names)
        self.relation_texts = [f"{r['src']} {r['rel']} {r['tgt']}" for r in self.relations]
        if self.relation_texts:
            self.relation_emb = index.encode(self.relation_texts)
        self._detect_communities(resolution)

    def _detect_communities(self, resolution: float) -> None:
        G = self.G
        if G.number_of_nodes() == 0:
            return
        try:
            import igraph as ig
            import leidenalg
            nodes = list(G.nodes)
            idx = {n: i for i, n in enumerate(nodes)}
            g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges])
            g.es["weight"] = [G[u][v]["weight"] for u, v in G.edges]
            part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                            resolution_parameter=resolution, seed=42)
            for cid, members in enumerate(part):
                for m in members:
                    self.communities[nodes[m]] = cid
            self.community_method = "leiden (leidenalg)"
        except Exception:
            comms = self.nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=42)
            for cid, members in enumerate(comms):
                for m in members:
                    self.communities[m] = cid
            self.community_method = "louvain (networkx fallback)"

    def _summarize(self, llm: BaseLLM) -> None:
        by_c: Dict[int, List[str]] = defaultdict(list)
        for n, c in self.communities.items():
            by_c[c].append(n)
        for c, members in by_c.items():
            if len(members) < 3:
                continue
            rels = [f"{r['src']} {r['rel']} {r['tgt']}" for r in self.relations if r["src"] in members and r["tgt"] in members][:40]
            prompt = COMMUNITY_SUMMARY.format(entities=", ".join(members[:60]), relations="\n".join(rels))
            self.summaries[c] = llm.call(prompt, stage="community_summary", json_mode=False, use_cache=True).text

    # ----------------------------------------------------------------- query-time helpers
    def match_entities(self, index: DenseIndex, texts: Sequence[str], top_m: int) -> List[str]:
        """Entities matched by (a) name occurrence in the text and (b) embedding similarity."""
        if not self.entity_names:
            return []
        found: List[str] = []
        joined = " ".join(norm_name(t) for t in texts)
        for name in self.entity_names:
            if len(name) >= 4 and (" " + name + " ") in (" " + joined + " "):
                found.append(name)
        q = index.embedder.encode_queries(list(texts))
        sims = self.entity_emb @ q.T                     # (n_entities, n_texts)
        for j in range(sims.shape[1]):
            for i in np.argsort(-sims[:, j], kind="stable")[:top_m]:
                found.append(self.entity_names[int(i)])
        return list(dict.fromkeys(found))

    def match_relations(self, index: DenseIndex, texts: Sequence[str], top_m: int) -> List[int]:
        if not self.relation_texts:
            return []
        q = index.embedder.encode_queries(list(texts))
        sims = self.relation_emb @ q.T
        out: List[int] = []
        for j in range(sims.shape[1]):
            out.extend(int(i) for i in np.argsort(-sims[:, j], kind="stable")[:top_m])
        return list(dict.fromkeys(out))

    def neighbourhood(self, entities: Sequence[str]) -> List[str]:
        out = list(entities)
        for e in entities:
            if e in self.G:
                out.extend(self.G.neighbors(e))
        return list(dict.fromkeys(out))

    def chunks_of_entities(self, entities: Sequence[str]) -> Set[int]:
        out: Set[int] = set()
        for e in entities:
            out |= self.entity_chunks.get(e, set())
        return out

    def chunks_of_relations(self, rel_ids: Sequence[int]) -> Set[int]:
        out: Set[int] = set()
        for i in rel_ids:
            r = self.relations[i]
            out |= r["chunks"]
            out |= self.entity_chunks.get(r["src"], set()) | self.entity_chunks.get(r["tgt"], set())
        return out

    def save_stats(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)


def rank_candidates(index: DenseIndex, query: str, candidates: Set[int], pool_k: int, min_candidates: int) -> Tuple[Ranked, Dict]:
    """Rank graph candidates by dense similarity; pad with global dense results so that
    doc-level metrics always see ``pool_k`` chunks.  Returns (ranking, diagnostics)."""
    info = {"n_candidates": len(candidates), "fallback": False}
    ranked: Ranked = []
    if len(candidates) >= min_candidates:
        ranked = index.score_subset(query, sorted(candidates))
    else:
        info["fallback"] = True
    if len(ranked) < pool_k:
        have = {i for i, _ in ranked}
        for i, s in index.search(query, pool_k * 2):
            if i not in have:
                ranked.append((i, s - 10.0))         # padded items always rank below graph candidates
                have.add(i)
            if len(ranked) >= pool_k:
                break
    return ranked[:pool_k], info
