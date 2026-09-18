"""Per-query metrics (binary relevance).

Retrieval metrics are computed on the ranking that is passed to the generator
(document level = unique documents in order of first appearance; sentence level = chunks).
Evidence metrics compare the *cited* sentence ids in the JSON output with the gold rationale union.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set


def precision_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    top = list(ranked[:k])
    return sum(1 for x in top if x in gold) / k if k else 0.0


def recall_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return sum(1 for x in ranked[:k] if x in gold) / len(gold)


def ndcg_at_k(ranked: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def mrr(ranked: Sequence[str], gold: Set[str]) -> float:
    if not gold:
        return float("nan")
    for i, x in enumerate(ranked):
        if x in gold:
            return 1.0 / (i + 1)
    return 0.0


def evidence_prf(cited: Sequence[str], gold: Set[str]) -> Dict[str, float]:
    cited_set = set(cited)
    if not gold:
        return {"evidence_precision": float("nan"), "evidence_recall": float("nan"), "evidence_f1": float("nan")}
    tp = len(cited_set & gold)
    p = tp / len(cited_set) if cited_set else 0.0
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"evidence_precision": p, "evidence_recall": r, "evidence_f1": f}


def full_rationale_recovered(cited: Sequence[str], gold_sets: Sequence[Sequence[str]]) -> float:
    c = set(cited)
    return 1.0 if any(set(g) <= c and g for g in gold_sets) else 0.0


def query_metrics(doc_ranking: List[str], sent_ranking: List[str], claim, gen, k: int) -> Dict[str, float]:
    gold_docs, gold_sents = set(claim.gold_doc_ids), set(claim.gold_sentence_ids)
    m = {
        "doc_precision@k": precision_at_k(doc_ranking, gold_docs, k) if gold_docs else float("nan"),
        "doc_recall@k": recall_at_k(doc_ranking, gold_docs, k),
        "doc_ndcg@k": ndcg_at_k(doc_ranking, gold_docs, k),
        "doc_mrr": mrr(doc_ranking, gold_docs),
        "sent_precision@k": precision_at_k(sent_ranking, gold_sents, k) if gold_sents else float("nan"),
        "sent_recall@k": recall_at_k(sent_ranking, gold_sents, k),
        "sent_ndcg@k": ndcg_at_k(sent_ranking, gold_sents, k),
        "sent_mrr": mrr(sent_ranking, gold_sents),
    }
    m.update(evidence_prf(gen.cited, gold_sents))
    m["full_rationale_recovered"] = full_rationale_recovered(gen.cited, claim.gold_evidence_sets) if gold_sents else float("nan")
    m["verdict_correct"] = 1.0 if gen.label == claim.label else 0.0
    m["fever_score"] = (m["verdict_correct"] * m["full_rationale_recovered"]) if gold_sents else m["verdict_correct"]
    m["json_valid"] = 1.0 if gen.json_valid else 0.0
    return m
