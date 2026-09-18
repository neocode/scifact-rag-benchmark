"""SciFact loading, benchmark construction, sentence chunking and identifier export."""
from __future__ import annotations

import json
import os
import random
import re
import tarfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LABEL_MAP = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}


@dataclass
class Claim:
    qid: str
    text: str
    label: str                                   # SUPPORTS | REFUTES | NOT_ENOUGH_INFO
    gold_doc_ids: List[str]
    gold_sentence_ids: List[str]                 # chunk ids "doc::sent_i" (union over evidence abstracts)
    gold_evidence_sets: List[List[str]]          # disjunctive rationale sets
    cited_doc_ids: List[str] = field(default_factory=list)


@dataclass
class Doc:
    doc_id: str
    title: str
    sentences: List[str]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    sent_id: int
    text: str
    title: str

    def as_dict(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------------ download / load
def ensure_download(data_dir: str) -> Path:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "data.tar.gz"
    extract_dir = root / "extracted"
    if not any(extract_dir.rglob("corpus.jsonl")):
        if not tar_path.exists() or tar_path.stat().st_size < 1_000_000:
            import requests
            print(f"Downloading SciFact release to {tar_path} ...")
            with requests.get(SCIFACT_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tar_path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            for m in tar.getmembers():
                if ".." in m.name or m.name.startswith("/"):
                    raise RuntimeError(f"suspicious tar member {m.name}")
            tar.extractall(extract_dir)
    corpus = next(extract_dir.rglob("corpus.jsonl"))
    return corpus.parent


def _read_jsonl(p: Path) -> List[Dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_corpus(base: Path) -> Dict[str, Doc]:
    docs = {}
    for r in _read_jsonl(base / "corpus.jsonl"):
        did = str(r["doc_id"])
        docs[did] = Doc(doc_id=did, title=str(r.get("title", "")), sentences=[str(s).strip() for s in r.get("abstract", [])])
    return docs


def load_claims(base: Path, split: str, docs: Dict[str, Doc]) -> List[Claim]:
    out = []
    for r in _read_jsonl(base / f"claims_{split}.jsonl"):
        ev = r.get("evidence") or {}
        labels, gold_docs, gold_sets, union = set(), [], [], set()
        for did, items in ev.items():
            did = str(did)
            if did not in docs:
                continue
            gold_docs.append(did)
            for it in items:
                lab = LABEL_MAP.get(str(it.get("label", "")).upper())
                if lab:
                    labels.add(lab)
                sids = sorted(int(s) for s in it.get("sentences", []))
                sset = [f"{did}::sent_{s}" for s in sids]
                if sset:
                    gold_sets.append(sset)
                    union.update(sset)
        if len(labels) == 1:
            label = labels.pop()
        elif len(labels) == 0:
            label = "NOT_ENOUGH_INFO"
        else:                       # conflicting evidence labels (rare) -> excluded downstream
            label = "CONFLICT"
        out.append(Claim(qid=str(r["id"]), text=str(r["claim"]).strip(), label=label,
                         gold_doc_ids=sorted(set(gold_docs)), gold_sentence_ids=sorted(union),
                         gold_evidence_sets=gold_sets, cited_doc_ids=[str(x) for x in r.get("cited_doc_ids", [])]))
    return out


# ------------------------------------------------------------------ benchmark construction
_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("the and for with that this from are was were has have had into using use used can may might also between within across of in on to by an a is as at or be we our these those their its it not no than more less most".split())


def _kw(text: str, n: int) -> Set[str]:
    toks = [t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 2]
    return set(w for w, _ in Counter(toks).most_common(n))


def build_main_benchmark(docs: Dict[str, Doc], claims: List[Claim], n_claims: int, n_docs: int,
                         hard_negative_pool: int, seed: int) -> Tuple[List[Claim], Dict[str, Doc], Dict]:
    """Fixed subset of the train split: SUPPORTS/REFUTES claims with annotated evidence.

    Corpus = all evidence abstracts of the sampled claims + keyword-overlap hard negatives
    (sampled with the same seed) up to ``n_docs``.
    """
    rng = random.Random(seed)
    eligible = [c for c in claims if c.label in ("SUPPORTS", "REFUTES") and c.gold_evidence_sets]
    eligible.sort(key=lambda c: int(c.qid))
    sampled = sorted(rng.sample(eligible, min(n_claims, len(eligible))), key=lambda c: int(c.qid))

    required: Set[str] = set()
    for c in sampled:
        required |= set(c.gold_doc_ids)
    claim_kw = [_kw(c.text, 20) for c in sampled]
    scored = []
    for did, d in docs.items():
        if did in required:
            continue
        dkw = _kw(d.title + " " + " ".join(d.sentences), 40)
        ov = max((len(dkw & ck) for ck in claim_kw), default=0)
        if ov > 0:
            scored.append((did, ov))
    scored.sort(key=lambda x: (-x[1], int(x[0])))
    pool = [d for d, _ in scored[:hard_negative_pool]]
    need = max(0, n_docs - len(required))
    negatives = sorted(rng.sample(pool, min(need, len(pool))), key=int)
    corpus = {did: docs[did] for did in sorted(required | set(negatives), key=int)}
    info = {
        "split": "train", "seed": seed, "n_claims": len(sampled), "n_evidence_docs": len(required),
        "n_hard_negatives": len(negatives), "n_docs": len(corpus),
        "label_distribution": dict(Counter(c.label for c in sampled)),
        "claims_with_multiple_evidence_docs": sum(1 for c in sampled if len(c.gold_doc_ids) > 1),
        "gold_sentences_per_claim_mean": round(sum(len(c.gold_sentence_ids) for c in sampled) / max(1, len(sampled)), 3),
    }
    return sampled, corpus, info


def build_dev_benchmark(docs: Dict[str, Doc], claims: List[Claim]) -> Tuple[List[Claim], Dict[str, Doc], Dict]:
    """Standard open SciFact dev split: all claims (incl. NOT_ENOUGH_INFO) over the full corpus."""
    kept = [c for c in claims if c.label != "CONFLICT"]
    info = {
        "split": "dev", "n_claims": len(kept), "n_docs": len(docs),
        "excluded_conflicting_label_claims": len(claims) - len(kept),
        "label_distribution": dict(Counter(c.label for c in kept)),
        "claims_with_evidence": sum(1 for c in kept if c.gold_evidence_sets),
    }
    return kept, docs, info


def make_chunks(corpus: Dict[str, Doc]) -> List[Chunk]:
    chunks = []
    for did in sorted(corpus, key=int):
        d = corpus[did]
        for sid, s in enumerate(d.sentences):
            if s:
                chunks.append(Chunk(chunk_id=f"{did}::sent_{sid}", doc_id=did, sent_id=sid, text=s, title=d.title))
    return chunks


def export_benchmark(path_dir: str, claims: List[Claim], corpus: Dict[str, Doc], chunks: List[Chunk], info: Dict) -> None:
    os.makedirs(path_dir, exist_ok=True)
    with open(os.path.join(path_dir, "benchmark_claims.jsonl"), "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    with open(os.path.join(path_dir, "benchmark_doc_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(corpus, key=int)) + "\n")
    with open(os.path.join(path_dir, "benchmark_chunk_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(c.chunk_id for c in chunks) + "\n")
    info = dict(info, n_chunks=len(chunks))
    with open(os.path.join(path_dir, "benchmark_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def overlap_report(main_claims: List[Claim], other_claims: List[Claim]) -> Dict:
    a = {c.qid for c in main_claims}
    b = {c.qid for c in other_claims}
    return {"shared_claim_ids": sorted(a & b, key=int), "n_shared": len(a & b)}
