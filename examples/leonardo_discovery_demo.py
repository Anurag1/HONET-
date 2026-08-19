"""Small deterministic end-to-end demonstration of the discovery loop."""

from leonardo_ai import DiscoveryEngine, Observation


def main() -> None:
    engine = DiscoveryEngine()

    # Observe first; no explanation is supplied with the observations.
    engine.observe([
        Observation("measurement_1", {"x": 1, "y": 3}, source="synthetic"),
        Observation("measurement_2", {"x": 2, "y": 5}, source="synthetic"),
        Observation("measurement_3", {"x": 4, "y": 9}, source="synthetic"),
    ])

    print("QUESTIONS")
    for question in engine.question():
        print(f"- {question}")

    # Competing hypotheses make the falsification step explicit.
    h1 = engine.hypothesize("y=2x+1", lambda x: 2 * x + 1)
    h2 = engine.hypothesize("y=3x", lambda x: 3 * x)

    for hypothesis in (h1, h2):
        result = engine.experiment(hypothesis, input_value=7, observed=15)
        print(
            f"{hypothesis.name}: predicted={result.predicted}, "
            f"observed={result.observed}, status={result.status}, error={result.error}"
        )

    print("REVISION")
    print(engine.revise(h2))
    print("GENERALIZATION")
    print(engine.generalize())


if __name__ == "__main__":
    main()
