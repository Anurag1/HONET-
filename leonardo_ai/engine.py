"""Executable observation -> question -> hypothesis -> experiment loop.

The implementation is intentionally model-agnostic and dependency-free so the
reasoning loop can be benchmarked independently of an LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from statistics import mean
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class Observation:
    """A recorded fact with provenance and confidence."""

    subject: str
    value: Any
    source: str = "unknown"
    confidence: float = 1.0
    tags: tuple[str, ...] = ()


@dataclass
class Hypothesis:
    """A falsifiable explanation represented by a predictor."""

    name: str
    predictor: Callable[[Any], Any]
    assumptions: list[str] = field(default_factory=list)
    evidence: list[Observation] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentResult:
    """Prediction versus observed result."""

    hypothesis: str
    predicted: Any
    observed: Any
    error: float
    status: str


class DiscoveryEngine:
    """A compact Leonardo-inspired scientific reasoning controller.

    The engine deliberately separates observation from explanation. It never
    treats a hypothesis as knowledge until an experiment produces evidence.
    """

    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.hypotheses: list[Hypothesis] = []
        self.experiments: list[ExperimentResult] = []

    def observe(self, observations: Iterable[Observation]) -> list[Observation]:
        """Record observations without interpreting them."""
        items = list(observations)
        if not items:
            raise ValueError("at least one observation is required")
        for item in items:
            if not 0.0 <= item.confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
        self.observations.extend(items)
        return items

    def decompose(self, value: Any) -> dict[str, Any]:
        """Expose simple structural components for downstream reasoning."""
        if isinstance(value, dict):
            return {"type": "mapping", "keys": list(value), "values": list(value.values())}
        if isinstance(value, (list, tuple)):
            return {"type": "sequence", "length": len(value), "items": list(value)}
        return {"type": type(value).__name__, "value": value}

    def measure(self, values: Sequence[float]) -> dict[str, float]:
        """Calculate basic descriptive measurements."""
        if not values:
            raise ValueError("values must not be empty")
        return {
            "count": float(len(values)),
            "mean": mean(values),
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
        }

    def connect(self, left: Sequence[Any], right: Sequence[Any]) -> list[tuple[Any, Any]]:
        """Create explicit cross-domain pairings rather than vague similarity."""
        return list(zip(left, right))

    def question(self) -> list[str]:
        """Generate falsification-oriented questions from current evidence."""
        if not self.observations:
            return ["What should be observed before an explanation is proposed?"]
        questions = [
            "Which observation is unexplained?",
            "Which assumption could make the explanation fail?",
            "What measurement would distinguish competing hypotheses?",
            "What result would falsify the leading hypothesis?",
        ]
        if any(o.confidence < 0.8 for o in self.observations):
            questions.append("Which low-confidence observation should be independently verified?")
        return questions

    def hypothesize(
        self,
        name: str,
        predictor: Callable[[Any], Any],
        assumptions: Iterable[str] = (),
    ) -> Hypothesis:
        """Register a falsifiable hypothesis."""
        hypothesis = Hypothesis(name=name, predictor=predictor, assumptions=list(assumptions))
        hypothesis.evidence.extend(self.observations)
        self.hypotheses.append(hypothesis)
        return hypothesis

    def predict(self, hypothesis: Hypothesis, input_value: Any) -> Any:
        return hypothesis.predictor(input_value)

    def experiment(
        self,
        hypothesis: Hypothesis,
        input_value: Any,
        observed: Any,
        tolerance: float = 1e-9,
    ) -> ExperimentResult:
        """Compare a pre-registered prediction with an observed result."""
        predicted = self.predict(hypothesis, input_value)
        try:
            error = abs(float(predicted) - float(observed))
            status = "CONFIRMED" if isclose(float(predicted), float(observed), abs_tol=tolerance) else "FALSIFIED"
        except (TypeError, ValueError):
            error = 0.0 if predicted == observed else 1.0
            status = "CONFIRMED" if predicted == observed else "FALSIFIED"

        result = ExperimentResult(hypothesis.name, predicted, observed, error, status)
        self.experiments.append(result)
        if status == "FALSIFIED":
            hypothesis.contradictions.append(f"predicted={predicted!r}, observed={observed!r}")
        return result

    def revise(self, hypothesis: Hypothesis) -> dict[str, Any]:
        """Return explicit revision evidence rather than silently changing a model."""
        related = [e for e in self.experiments if e.hypothesis == hypothesis.name]
        failures = [e for e in related if e.status == "FALSIFIED"]
        return {
            "hypothesis": hypothesis.name,
            "experiments": len(related),
            "failures": len(failures),
            "revision_required": bool(failures),
            "contradictions": list(hypothesis.contradictions),
        }

    def generalize(self) -> dict[str, Any]:
        """Summarize validated versus rejected hypotheses."""
        confirmed = [e.hypothesis for e in self.experiments if e.status == "CONFIRMED"]
        falsified = [e.hypothesis for e in self.experiments if e.status == "FALSIFIED"]
        return {
            "observations": len(self.observations),
            "experiments": len(self.experiments),
            "confirmed": confirmed,
            "falsified": falsified,
            "next_cycle": "OBSERVE",
        }
