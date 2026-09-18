"""The compared pipelines.

Every pipeline exposes ``retrieve(claim_text) -> (ranking, diagnostics)`` where ``ranking`` is a
list of (chunk_index, score) of length ``pool_k``; the first ``k`` items form the generation context.
Generation is identical for all pipelines (``Generator``), so differences stem from retrieval /
orchestration only.  All corrective and agentic actions are restricted to the indexed corpus
(no web search) - the CRAG variant is therefore a *corpus-restricted* CRAG.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .llm import BaseLLM, parse_json_lenient, parse_json_strict
from .prompts import (AGENT_CONTROLLER, CRAG_EVALUATOR, GENERATION_2CLASS, GENERATION_3CLASS,
                      LIGHTRAG_KEYWORDS, QUERY_REFORMULATION, format_evidence)
from .retrieval import BM25Index, DenseIndex, Ranked, rrf
from .graph import KnowledgeGraph, rank_candidates


@dataclass
class Resources:
    cfg: object
    chunks: List[Dict]                 # dicts with chunk_id, doc_id, sent_id, text
    dense: DenseIndex
    bm25: BM25Index
    llm: BaseLLM
    reranker: object = None
    kg: Optional[KnowledgeGraph] = None

    @property
    def texts(self) -> List[str]:
        return [c["text"] for c in self.chunks]


class System:
    name = "System"
    uses_llm_in_retrieval = False

    def __init__(self, res: Resources):
        self.res = res
        self.cfg = res.cfg

    def retrieve(self, claim: str) -> Tuple[Ranked, Dict]:
        raise NotImplementedError


# --------------------------------------------------------------------------- baselines
class BM25System(System):
    name = "BM25"

    def retrieve(self, claim):
        return self.res.bm25.search(claim, self.cfg.pool_k), {}


class DenseSystem(System):
    name = "Dense"

    def retrieve(self, claim):
        return self.res.dense.search(claim, self.cfg.pool_k), {}


class HybridSystem(System):
    name = "Hybrid"

    def retrieve(self, claim):
        d = self.res.dense.search(claim, self.cfg.pool_k)
        b = self.res.bm25.search(claim, self.cfg.pool_k)
        return rrf([d, b], self.cfg.rrf_k)[: self.cfg.pool_k], {}


class RerankSystem(System):
    name = "Dense+Rerank"

    def retrieve(self, claim):
        pool = self.res.dense.search(claim, self.cfg.pool_k)
        head = self.res.reranker.rerank(claim, pool[: self.cfg.rerank_depth], self.res.texts, self.cfg.rerank_depth)
        seen = {i for i, _ in head}
        tail = [(i, s - 100.0) for i, s in pool if i not in seen]
        return (head + tail)[: self.cfg.pool_k], {}


# --------------------------------------------------------------------------- CRAG (corpus-restricted)
class CRAGSystem(System):
    name = "CRAG"
    uses_llm_in_retrieval = True

    def _reformulate(self, claim: str) -> str:
        r = self.res.llm.call(QUERY_REFORMULATION.format(claim=claim), stage="crag_reformulate")
        obj = parse_json_lenient(r.text)
        q = obj.get("query") if isinstance(obj, dict) else None
        return str(q).strip() if q else claim

    def retrieve(self, claim):
        cfg, res = self.cfg, self.res
        k = cfg.k
        base = res.dense.search(claim, cfg.pool_k)
        top = [res.chunks[i] for i, _ in base[:k]]
        r = res.llm.call(CRAG_EVALUATOR.format(claim=claim, evidence=format_evidence(top)), stage="crag_evaluate")
        obj = parse_json_lenient(r.text)
        scores: Dict[str, float] = {}
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            for cid, v in obj["scores"].items():
                try:
                    scores[str(cid)] = min(1.0, max(0.0, float(v)))
                except Exception:
                    continue
        vals = [scores.get(c["chunk_id"], 0.0) for c in top]
        mx = max(vals) if vals else 0.0
        mean = sum(vals) / len(vals) if vals else 0.0
        if mx > cfg.crag_theta_high and mean > cfg.crag_theta_mean:
            branch = "correct"
            ranking = base
        elif mx < cfg.crag_theta_low:
            branch = "incorrect"          # web search disabled -> corpus-restricted substitute: reformulate + hybrid
            q2 = self._reformulate(claim)
            ranking = rrf([base, res.dense.search(q2, cfg.pool_k), res.bm25.search(q2, cfg.pool_k)], cfg.rrf_k)
        else:
            branch = "ambiguous"          # expanded retrieval with query reformulation
            q2 = self._reformulate(claim)
            ranking = rrf([base, res.dense.search(q2, cfg.pool_k)], cfg.rrf_k)
        # knowledge refinement: demote evaluated chunks judged irrelevant
        low = {res.chunks[i]["chunk_id"] for i, _ in base[:k] if scores.get(res.chunks[i]["chunk_id"], 1.0) < cfg.crag_keep_threshold}
        if low and branch != "correct":
            keep = [(i, s) for i, s in ranking if res.chunks[i]["chunk_id"] not in low]
            drop = [(i, s - 100.0) for i, s in ranking if res.chunks[i]["chunk_id"] in low]
            ranking = keep + drop
        info = {"crag_branch": branch, "eval_max": round(mx, 3), "eval_mean": round(mean, 3),
                "eval_parsed": bool(scores), "n_refined_out": len(low) if branch != "correct" else 0}
        return ranking[: cfg.pool_k], info


# --------------------------------------------------------------------------- Agentic RAG
class AgenticSystem(System):
    name = "Agentic"
    uses_llm_in_retrieval = True

    def retrieve(self, claim):
        cfg, res = self.cfg, self.res
        lists: List[Ranked] = [res.dense.search(claim, cfg.agent_pool_per_step)]
        history: List[str] = [claim]
        actions: List[Dict] = []
        iterations = 0
        finished_by = "iteration_limit"
        for it in range(1, cfg.agent_max_iterations + 1):
            fused = rrf(lists, cfg.rrf_k)
            ev = format_evidence([res.chunks[i] for i, _ in fused[: cfg.k]])
            hist = "\n".join(f"{n}. {h}" for n, h in enumerate(history, 1))
            prompt = AGENT_CONTROLLER.format(remaining=cfg.agent_max_iterations - it + 1, claim=claim, history=hist, evidence=ev)
            r = res.llm.call(prompt, stage="agent_controller")
            iterations = it
            obj = parse_json_lenient(r.text)
            if not isinstance(obj, dict):
                actions.append({"action": "invalid"})
                finished_by = "invalid_controller_output"
                break
            act = str(obj.get("action", "")).lower()
            actions.append({"action": act, "query": obj.get("query", ""), "thought": str(obj.get("thought", ""))[:200]})
            if act == "search" and str(obj.get("query", "")).strip():
                q = str(obj["query"]).strip()
                history.append(q)
                lists.append(res.dense.search(q, cfg.agent_pool_per_step))
            else:
                finished_by = "agent_finish"
                break
        ranking = rrf(lists, cfg.rrf_k)
        if len(ranking) < cfg.pool_k:                   # pad for doc-level metrics
            have = {i for i, _ in ranking}
            ranking += [(i, s - 100.0) for i, s in res.dense.search(claim, cfg.pool_k) if i not in have]
        info = {"agent_iterations": iterations, "agent_searches": len(history) - 1, "agent_finished_by": finished_by,
                "agent_actions": actions}
        return ranking[: cfg.pool_k], info


# --------------------------------------------------------------------------- Graph-based
class GraphRAGSystem(System):
    """GraphRAG local search: query entities -> 1-hop neighbourhood -> provenance chunks -> dense re-ranking."""
    name = "GraphRAG"

    def retrieve(self, claim):
        cfg, res = self.cfg, self.res
        ents = res.kg.match_entities(res.dense, [claim], cfg.graph_entity_top_m)
        hood = res.kg.neighbourhood(ents)
        cand = res.kg.chunks_of_entities(hood)
        ranking, info = rank_candidates(res.dense, claim, cand, cfg.pool_k, cfg.graph_min_candidates)
        info.update({"n_query_entities": len(ents), "n_neighbourhood_entities": len(hood)})
        return ranking, info


class LightRAGSystem(System):
    """LightRAG dual-level retrieval: low-level (entity) + high-level (relation) keyword matching."""
    name = "LightRAG"
    uses_llm_in_retrieval = True

    def retrieve(self, claim):
        cfg, res = self.cfg, self.res
        r = res.llm.call(LIGHTRAG_KEYWORDS.format(claim=claim), stage="lightrag_keywords")
        obj = parse_json_lenient(r.text) or {}
        low = [str(x) for x in (obj.get("low_level") or []) if str(x).strip()] or [claim]
        high = [str(x) for x in (obj.get("high_level") or []) if str(x).strip()] or [claim]
        ents = res.kg.match_entities(res.dense, low, cfg.graph_entity_top_m)
        rels = res.kg.match_relations(res.dense, high, cfg.graph_relation_top_m)
        cand = res.kg.chunks_of_entities(ents) | res.kg.chunks_of_relations(rels)
        ranking, info = rank_candidates(res.dense, claim, cand, cfg.pool_k, cfg.graph_min_candidates)
        info.update({"n_low_keywords": len(low), "n_high_keywords": len(high), "n_entities": len(ents), "n_relations": len(rels)})
        return ranking, info


SYSTEMS = {
    "BM25": BM25System, "Dense": DenseSystem, "Hybrid": HybridSystem, "Dense+Rerank": RerankSystem,
    "CRAG": CRAGSystem, "Agentic": AgenticSystem, "GraphRAG": GraphRAGSystem, "LightRAG": LightRAGSystem,
}


# --------------------------------------------------------------------------- shared generation
@dataclass
class Generation:
    label: str
    cited: List[str]
    json_valid: bool
    raw: str
    answer: str = ""


class Generator:
    def __init__(self, llm: BaseLLM, labels: List[str]):
        self.llm = llm
        self.labels = labels
        self.template = GENERATION_3CLASS if "NOT_ENOUGH_INFO" in labels else GENERATION_2CLASS

    def generate(self, claim: str, context: List[Dict]) -> Generation:
        allowed = [c["chunk_id"] for c in context]
        r = self.llm.call(self.template.format(claim=claim, evidence=format_evidence(context)), stage="generation")
        obj = parse_json_strict(r.text)
        valid = isinstance(obj, dict) and str(obj.get("label", "")).strip().upper() in self.labels \
            and isinstance(obj.get("evidence", []), list)
        if not valid:
            return Generation(label="INVALID", cited=[], json_valid=False, raw=r.text)
        cited = [str(x) for x in obj.get("evidence", []) if str(x) in allowed]
        return Generation(label=str(obj["label"]).strip().upper(), cited=list(dict.fromkeys(cited)),
                          json_valid=True, raw=r.text, answer=str(obj.get("answer", ""))[:1000])
