"""LLM access: Gemini client with retries, token accounting and an offline mock.

Every call is recorded in the active ``CallLog`` so that the number of model calls,
token usage and LLM wall time can be reported per query and per pipeline stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_ms: float = 0.0
    model_version: str = ""
    cached: bool = False


@dataclass
class CallLog:
    """Accumulates LLM usage for one query (reset by the benchmark runner)."""
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    llm_ms: float = 0.0
    stages: Dict[str, int] = field(default_factory=dict)
    model_version: str = ""

    def add(self, stage: str, r: LLMResponse) -> None:
        self.calls += 1
        if r.model_version:
            self.model_version = r.model_version
        self.prompt_tokens += r.prompt_tokens
        self.output_tokens += r.output_tokens
        self.thinking_tokens += r.thinking_tokens
        self.llm_ms += r.latency_ms
        self.stages[stage] = self.stages.get(stage, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "llm_ms": round(self.llm_ms, 1),
            "calls_by_stage": dict(self.stages),
            "model_version": self.model_version,
        }


# ------------------------------------------------------------------ JSON helpers
def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def parse_json_strict(s: str) -> Optional[Any]:
    """Strict parse: the whole response (after fence stripping) must be JSON."""
    try:
        return json.loads(strip_code_fences(s))
    except Exception:
        return None


def parse_json_lenient(s: str) -> Optional[Any]:
    """Lenient parse used for *orchestration* calls only (evaluator, controller, extraction)."""
    obj = parse_json_strict(s)
    if obj is not None:
        return obj
    s = strip_code_fences(s)
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except Exception:
            pass
    return None


# ------------------------------------------------------------------ disk cache (offline stages only)
class DiskCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._d: Dict[str, str] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._d[rec["k"]] = rec["v"]
                    except Exception:
                        continue

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()

    def get(self, k: str) -> Optional[str]:
        return self._d.get(k)

    def put(self, k: str, v: str) -> None:
        self._d[k] = v
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ clients
class BaseLLM:
    name = "base"

    def __init__(self, cache: Optional[DiskCache] = None, pause_s: float = 0.0):
        self.cache = cache
        self.pause_s = pause_s
        self.log: CallLog = CallLog()
        self.total_calls = 0

    def reset_log(self) -> None:
        self.log = CallLog()

    def call(self, prompt: str, stage: str, json_mode: bool = True, use_cache: bool = False) -> LLMResponse:
        if use_cache and self.cache is not None:
            k = DiskCache.key(self.name, prompt)
            hit = self.cache.get(k)
            if hit is not None:
                r = LLMResponse(text=hit, cached=True)
                self.log.add(stage, r)
                return r
        r = self._call(prompt, json_mode)
        if use_cache and self.cache is not None:
            self.cache.put(DiskCache.key(self.name, prompt), r.text)
        self.log.add(stage, r)
        self.total_calls += 1
        if self.pause_s > 0:
            time.sleep(self.pause_s)
        return r

    def _call(self, prompt: str, json_mode: bool) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 8,
                 api_key_env: str = "GEMINI_API_KEY", thinking_budget: int = 0, **kw):
        super().__init__(**kw)
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Set the {api_key_env} environment variable (never hard-code the key).")
        from google import genai  # google-genai SDK
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"
        self.temperature = temperature
        self.max_retries = max_retries
        self.thinking_budget = thinking_budget

    def _call(self, prompt: str, json_mode: bool) -> LLMResponse:
        kwargs = dict(temperature=self.temperature,
                      response_mime_type="application/json" if json_mode else "text/plain")
        if self.thinking_budget is not None and self.thinking_budget >= 0:
            # Gemini 2.5 models think by default (dynamic budget). A fixed budget (0 = off) keeps the
            # generator identical across pipelines and keeps latency/cost free of hidden reasoning tokens.
            try:
                kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=int(self.thinking_budget))
            except Exception:
                pass  # older SDK / model without thinking support
        cfg = self._types.GenerateContentConfig(**kwargs)
        last = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                ms = (time.perf_counter() - t0) * 1000.0
                um = getattr(resp, "usage_metadata", None)
                try:
                    text = resp.text or ""          # None / ValueError when the response was blocked
                except Exception:
                    text = ""
                return LLMResponse(
                    text=text.strip(),
                    prompt_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
                    output_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
                    thinking_tokens=int(getattr(um, "thoughts_token_count", 0) or 0),
                    latency_ms=ms,
                    model_version=str(getattr(resp, "model_version", "") or self.model),
                )
            except Exception as e:  # transient errors -> exponential backoff
                last = e
                msg = str(e).upper()
                transient = any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "RATE", "503", "500", "502", "504",
                                                   "DEADLINE", "UNAVAILABLE", "OVERLOADED", "TIMEOUT"))
                if not transient:
                    raise
                time.sleep(min(60.0, 2.0 * (2 ** attempt) + random.random()))
        raise RuntimeError(f"Gemini call failed after {self.max_retries} retries: {last}")


class MockLLM(BaseLLM):
    """Deterministic offline stand-in used to test the pipeline logic without API access."""
    name = "mock"

    def __init__(self, seed: int = 0, **kw):
        super().__init__(**kw)
        self.rng = random.Random(seed)

    def _call(self, prompt: str, json_mode: bool) -> LLMResponse:
        ids = re.findall(r"^\[([^\]\n]+)\]", prompt, flags=re.MULTILINE)
        ids = [i for i in ids if "::" in i]
        if "Task: scientific claim verification" in prompt:
            labels = ["SUPPORTS", "REFUTES"] + (["NOT_ENOUGH_INFO"] if "NOT_ENOUGH_INFO" in prompt else [])
            out = {"label": self.rng.choice(labels), "answer": "Mock answer.", "evidence": ids[:2]}
            if self.rng.random() < 0.02:
                return LLMResponse(text="not json at all")  # simulate a JSON failure
        elif "retrieval evaluator" in prompt:
            out = {"scores": {i: round(self.rng.random(), 2) for i in ids}}
        elif "Rewrite the following scientific claim" in prompt:
            m = re.search(r"Claim:\n(.*?)\n\nJSON", prompt, flags=re.S)
            out = {"query": (m.group(1) if m else "query") + " mechanism"}
        elif "controller of an iterative retrieval agent" in prompt:
            m = re.search(r"Claim:\n(.*?)\n\nSearch history", prompt, flags=re.S)
            if "2." in prompt.split("Search history:")[1].split("Evidence collected")[0]:
                out = {"action": "finish", "thought": "enough"}
            else:
                out = {"action": "search", "query": (m.group(1) if m else "q") + " evidence", "thought": "refine"}
        elif "Extract a knowledge graph" in prompt:
            body = prompt.split("Abstract sentences:")[1]
            sents = re.findall(r"^\[(\d+)\]\s*(.*)$", body, flags=re.MULTILINE)
            words = []
            for sid, s in sents:
                for w in re.findall(r"[A-Za-z][A-Za-z\-]{5,}", s):
                    words.append((w.lower(), int(sid)))
            uniq = list(dict.fromkeys(w for w, _ in words))[:8]
            ents = [{"name": w, "type": "concept"} for w in uniq]
            rels = []
            for i in range(len(uniq) - 1):
                sid = [s for w, s in words if w == uniq[i]][:1]
                rels.append({"source": uniq[i], "target": uniq[i + 1], "relation": "associated with", "sentence_ids": sid})
            out = {"entities": ents, "relations": rels}
        elif "Extract retrieval keywords" in prompt:
            m = re.search(r"Claim:\n(.*?)\n\nJSON", prompt, flags=re.S)
            toks = re.findall(r"[A-Za-z][A-Za-z\-]{5,}", m.group(1) if m else "")
            out = {"low_level": toks[:4], "high_level": toks[4:6] or ["association"]}
        elif "Summarise" in prompt:
            return LLMResponse(text="Mock community summary.", prompt_tokens=50, output_tokens=10, latency_ms=1.0)
        else:
            out = {}
        return LLMResponse(text=json.dumps(out), prompt_tokens=len(prompt.split()), output_tokens=20,
                           latency_ms=1.0 + self.rng.random(), model_version="mock")


def make_llm(cfg, cache_path: Optional[str] = None) -> BaseLLM:
    cache = DiskCache(cache_path) if cache_path else None
    if cfg.llm == "mock":
        return MockLLM(seed=cfg.seed, cache=cache, pause_s=0.0)
    return GeminiLLM(model=cfg.gemini_model, temperature=cfg.temperature, max_retries=cfg.max_retries,
                     thinking_budget=cfg.thinking_budget, cache=cache, pause_s=cfg.request_pause_s)
