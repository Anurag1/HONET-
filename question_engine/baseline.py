from __future__ import annotations

from dataclasses import dataclass

from question_engine.engine import CandidateQuestion, rank_questions


@dataclass(frozen=True)
class BenchmarkResult:
    baseline: list[CandidateQuestion]
    ranked: list[CandidateQuestion]


def compare(baseline: list[CandidateQuestion], generated: list[CandidateQuestion]) -> BenchmarkResult:
    """Compare a baseline candidate set with HONET's ranked candidate set."""
    return BenchmarkResult(baseline=baseline, ranked=rank_questions(generated))
