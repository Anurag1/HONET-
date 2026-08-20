"""Deterministic GBrain-style retrieval vs HONET discovery benchmark.

This is deliberately provider-independent: it models the two capabilities we
want to measure without pretending to embed Garry Tan's GBrain implementation.
Baseline = keyword retrieval + graph traversal + synthesis-like answer.
HONET = baseline plus contradiction/question/hypothesis generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import re
from typing import Iterable

from leonardo_ai.engine import DiscoveryEngine, Observation


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    source: str


FACTS = [
    Fact("AlphaAI", "hired", "Mira", "press_1"),
    Fact("Mira", "previously_at", "BetaLabs", "bio_1"),
    Fact("BetaLabs", "uses", "edge_inference", "tech_1"),
    Fact("AlphaAI", "uses", "cloud_inference", "tech_2"),
    Fact("AlphaAI", "claims", "lower_latency", "press_2"),
    Fact("BetaLabs", "claims", "lower_latency", "press_3"),
    Fact("AlphaAI", "latency_ms", "42", "bench_1"),
    Fact("BetaLabs", "latency_ms", "18", "bench_2"),
]


def keyword_retrieve(query: str, facts: Iterable[Fact], k: int = 5) -> list[Fact]:
    terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
    scored = []
    for fact in facts:
        text = " ".join((fact.subject, fact.relation, fact.object)).lower()
        score = sum(term in text for term in terms)
        if score:
            scored.append((score, fact))
    return [fact for _, fact in sorted(scored, key=lambda x: (-x[0], x[1].source))[:k]]


def graph_expand(seed: str, facts: Iterable[Fact]) -> list[Fact]:
    graph = defaultdict(list)
    for fact in facts:
        graph[fact.subject].append(fact)
        graph[fact.object].append(fact)
    return graph[seed]


def baseline(query: str) -> dict:
    retrieved = keyword_retrieve(query, FACTS)
    answer = " ".join(f"{f.subject} {f.relation} {f.object} [{f.source}]" for f in retrieved)
    return {"retrieved": retrieved, "answer": answer}


def honet_discovery(query: str) -> dict:
    base = baseline(query)
    engine = DiscoveryEngine()
    observations = [Observation(f"{f.subject}.{f.relation}", f.object, f.source) for f in base["retrieved"]]
    engine.observe(observations)

    # Explicitly test the contradiction: both firms claim lower latency, but
    # independently observed benchmark values differ.
    h = engine.hypothesize(
        "same-latency-claim",
        lambda _: "42",
        assumptions=["marketing claims are comparable", "latency metric is measured identically"],
    )
    result = engine.experiment(h, "BetaLabs", observed="18", tolerance=0.0)
    questions = engine.question()
    return {
        "baseline": base,
        "questions": questions,
        "hypothesis": h.name,
        "experiment": result,
        "revision": engine.revise(h),
    }


def main() -> None:
    query = "which company has lower latency and why?"
    b = baseline(query)
    d = honet_discovery(query)
    print("=== GBrain-style baseline ===")
    print(b["answer"])
    print("\n=== HONET discovery layer ===")
    print("Questions:")
    for q in d["questions"]:
        print("-", q)
    print("Hypothesis:", d["hypothesis"])
    print("Experiment:", d["experiment"])
    print("Revision:", d["revision"])


if __name__ == "__main__":
    main()
