# Discovery Engine Research Benchmark

This directory contains the preregistered experiment for testing whether explicit assumption + contradiction mapping improves AI research discovery.

## Research question
Does the contradiction-driven pipeline outperform a conventional literature-grounded research agent when model, evidence corpus, task set, and tool budgets are matched?

## Experimental arms

### A — Baseline
Conventional research agent: question -> evidence retrieval -> hypotheses -> answer.

### B — Discovery Engine
Observation extraction -> assumption graph -> contradiction/unknown detection -> competing hypotheses -> discriminating tests -> evidence retrieval -> ranking.

## Guardrails
- Freeze the benchmark questions before evaluation.
- Match model/version and resource budgets.
- Keep held-out questions untouched until final evaluation.
- Preserve all intermediate outputs.
- Do not call an idea a discovery without independent novelty/correctness validation.
- Report negative results.

## Primary outcomes
1. Novelty-adjusted hypothesis quality.
2. Falsifiability/testability rate.
3. Evidence grounding.

See `discovery_engine_preregistration.yaml` for the preregistered protocol.
