from __future__ import annotations

from dataclasses import dataclass
from math import log1p


@dataclass(frozen=True)
class CandidateQuestion:
    question: str
    novelty: float
    testability: float
    cross_domain: float
    information_gain: float

    @property
    def score(self) -> float:
        return round(
            0.30 * self.novelty
            + 0.25 * self.testability
            + 0.25 * self.cross_domain
            + 0.20 * self.information_gain,
            4,
        )


def information_gain(new_nodes: int, new_edges: int, contradictions_resolved: int) -> float:
    """Bounded heuristic used by the local benchmark; replace with empirical gain later."""
    raw = log1p(max(0, new_nodes) + 2 * max(0, new_edges) + 3 * max(0, contradictions_resolved))
    return min(1.0, raw / log1p(20))


def rank_questions(candidates: list[CandidateQuestion]) -> list[CandidateQuestion]:
    return sorted(candidates, key=lambda q: q.score, reverse=True)
