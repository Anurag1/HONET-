from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionScore:
    usefulness: float
    testability: float
    grounding: float
    cross_domain: float

    @property
    def total(self) -> float:
        return (
            0.35 * self.usefulness
            + 0.25 * self.testability
            + 0.20 * self.grounding
            + 0.20 * self.cross_domain
        )


def rank(score: QuestionScore) -> float:
    """Deterministic benchmark score used for baseline-vs-HONET evaluation."""
    return round(score.total, 4)
