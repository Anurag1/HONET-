# GBrain × HONET Discovery Benchmark

This experiment tests a narrow claim: a knowledge-brain retrieval layer can return evidence, while a HONET discovery layer can additionally turn retrieved evidence into falsifiable questions, hypotheses, experiments, and explicit revision signals.

It is **not** an implementation of GBrain itself. The baseline is a deterministic approximation of its relevant architectural capabilities: keyword retrieval plus relationship-aware evidence expansion. This keeps the benchmark reproducible and avoids attributing behavior to GBrain that this prototype does not execute.

## Experimental question

> Given the same evidence, does adding the HONET discovery loop produce an actionable falsification test that retrieval alone does not?

## Pipeline

```text
facts
  ├──> retrieval ──> evidence/answer
  │
  └──> retrieval ──> OBSERVE ─> QUESTION ─> HYPOTHESIZE
                                      │
                                      v
                                  PREDICT
                                      │
                                      v
                                  EXPERIMENT
                                      │
                             CONFIRMED/FALSIFIED
                                      │
                                      v
                                   REVISE
```

## Deterministic result

The fixture contains two companies that both claim lower latency, while independent benchmark observations report **42 ms** for AlphaAI and **18 ms** for BetaLabs.

The baseline can retrieve the claim and measurements. HONET additionally registers the claim as a falsifiable hypothesis, tests it against the observed 18 ms value, marks it **FALSIFIED**, and sets `revision_required=true`.

Run:

```bash
python examples/gbrain_honet_benchmark.py
python -m unittest tests/test_gbrain_honet_benchmark.py -v
```

## What this proves

- Retrieval can expose evidence.
- A discovery controller can explicitly separate observation from hypothesis.
- Contradictory evidence can trigger a falsification result rather than being silently averaged into an answer.
- The experiment is deterministic and unit-testable.

## What this does not prove

It does not prove that HONET discovers novel scientific knowledge, outperforms GBrain generally, or produces economically valuable discoveries. Those require a larger corpus, blinded baselines, independent evaluation, novelty checks, and human validation.
