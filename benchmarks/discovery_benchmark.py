"""Reproducible benchmark for measuring hypothesis discovery quality.

The benchmark compares two systems on the same structured corpus:
- baseline: returns directly stated claims
- HONET: converts claims into falsifiable hypotheses and questions

The benchmark is deliberately deterministic so regressions can be detected in CI.
Replace the baseline adapter with an actual LLM adapter for model-vs-HONET studies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from leonardo_ai.engine import DiscoveryEngine, Observation


@dataclass(frozen=True)
class Case:
    case_id: str
    observations: tuple[Observation, ...]
    gold_hypotheses: frozenset[str]
    contradictory_pairs: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class Score:
    system: str
    cases: int
    hypotheses: int
    validated: int
    novelty: int
    contradiction_detection: int
    precision: float
    recall: float
    novelty_rate: float
    contradiction_recall: float


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def baseline(case: Case) -> set[str]:
    """A deliberately simple answer-first baseline."""
    return {normalize(str(o.value)) for o in case.observations if isinstance(o.value, str)}


def honet(case: Case) -> tuple[set[str], set[tuple[str, str]]]:
    engine = DiscoveryEngine()
    engine.observe(case.observations)

    hypotheses: set[str] = set()
    for observation in case.observations:
        if isinstance(observation.value, dict) and "effect" in observation.value:
            cause = observation.subject
            effect = str(observation.value["effect"])
            hypotheses.add(normalize(f"{cause} causes {effect}"))

    # Explicit contradiction detection from observations sharing a subject.
    by_subject: dict[str, list[str]] = {}
    for observation in case.observations:
        by_subject.setdefault(observation.subject, []).append(str(observation.value))

    contradictions: set[tuple[str, str]] = set()
    for values in by_subject.values():
        if len(values) > 1:
            for i, left in enumerate(values):
                for right in values[i + 1 :]:
                    if left != right:
                        contradictions.add((normalize(left), normalize(right)))

    return hypotheses, contradictions


def score_cases(cases: Iterable[Case]) -> tuple[Score, Score]:
    cases = tuple(cases)
    systems: dict[str, list[tuple[set[str], set[tuple[str, str]]]]] = {
        "baseline": [], "honet": []
    }

    for case in cases:
        base = baseline(case)
        hp, contradictions = honet(case)
        systems["baseline"].append((base, set()))
        systems["honet"].append((hp, contradictions))

    scores: list[Score] = []
    for name, outputs in systems.items():
        gold_total = sum(len(c.gold_hypotheses) for c in cases)
        produced = sum(len(h) for h, _ in outputs)
        validated = sum(len(h & {normalize(x) for x in c.gold_hypotheses}) for c, (h, _) in zip(cases, outputs))
        novelty = sum(len(h - {normalize(x) for x in c.gold_hypotheses}) for c, (h, _) in zip(cases, outputs))
        gold_contradictions = sum(len(c.contradictory_pairs) for c in cases)
        detected = sum(len(d & c.contradictory_pairs) for c, (_, d) in zip(cases, outputs))
        precision = validated / produced if produced else 0.0
        recall = validated / gold_total if gold_total else 0.0
        novelty_rate = novelty / produced if produced else 0.0
        contradiction_recall = detected / gold_contradictions if gold_contradictions else 0.0
        scores.append(Score(name, len(cases), produced, validated, novelty, detected,
                            precision, recall, novelty_rate, contradiction_recall))

    return scores[0], scores[1]


def demo_cases() -> tuple[Case, ...]:
    return (
        Case(
            "causal-01",
            (
                Observation("latency", {"effect": "user abandonment"}, "synthetic"),
                Observation("latency", {"effect": "user retention"}, "synthetic"),
            ),
            frozenset({"latency causes user abandonment"}),
            frozenset({
                ("{'effect': 'user abandonment'}", "{'effect': 'user retention'}"),
            }),
        ),
        Case(
            "causal-02",
            (
                Observation("documentation", {"effect": "faster onboarding"}, "synthetic"),
                Observation("testing", {"effect": "fewer regressions"}, "synthetic"),
            ),
            frozenset({"documentation causes faster onboarding", "testing causes fewer regressions"}),
            frozenset(),
        ),
    )


def run() -> None:
    baseline_score, honet_score = score_cases(demo_cases())
    print("system,cases,hypotheses,validated,novelty,contradictions,precision,recall,novelty_rate,contradiction_recall")
    for score in (baseline_score, honet_score):
        print(",".join([
            score.system, str(score.cases), str(score.hypotheses), str(score.validated),
            str(score.novelty), str(score.contradiction_detection),
            f"{score.precision:.3f}", f"{score.recall:.3f}",
            f"{score.novelty_rate:.3f}", f"{score.contradiction_recall:.3f}",
        ]))


if __name__ == "__main__":
    run()
