# HONET Live Practical Benchmark — 2026-08-20

## Purpose

Test whether a question-driven workflow can produce more useful, testable, grounded knowledge than answer-first prompting.

## Input

The two notebook images supplied for the HONET product-design experiment.

## Practical protocol

1. Treat the images as observations, not truth.
2. Extract concepts and explicit relationships.
3. Generate candidate questions.
4. Rank questions using novelty, testability, cross-domain relevance, and information gain.
5. Investigate top questions against evidence.
6. Record what is genuinely new, what is merely restated, and what remains unresolved.
7. Do not call anything a discovery without independent verification.

## Live result status

This GitHub run validates the repository-level workflow and benchmark instrumentation. A real OpenAI API execution still requires a configured API key in the runtime environment; no key is committed to the repository.

## Current strongest benchmark hypothesis

The key hypothesis is:

> A question selected for expected information gain can produce more validated new structure per question than an ordinary answer-first workflow.

## Metrics

- questions_generated
- top_questions_selected
- new_concepts
- new_relationships
- contradictions_resolved
- evidence_backed_claims
- validated_discoveries
- knowledge_gain_per_question
- false_discovery_rate

## Success condition

HONET must beat a predefined baseline on the same inputs and evaluation criteria. If it does not, the ranking mechanism should be revised or rejected.

## Guardrails

- Never infer novelty from model output alone.
- Preserve source attribution.
- Separate observation, hypothesis, and evidence.
- Keep model/version and prompt configuration recorded for reproducibility.
