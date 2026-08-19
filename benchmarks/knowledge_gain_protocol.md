# HONET Knowledge Gain Benchmark

## Research question
Can a question-driven AI workflow produce more validated new knowledge per question than answer-first prompting?

## Inputs
Each benchmark case contains:
- source material (image, document, code, or text)
- extracted observations
- source identifiers
- optional domain label

The first case is the two-page handwritten notebook seed used during development.

## Systems
### Baseline
A standard LLM receives the same source material and is asked to identify useful insights and questions.

### HONET
1. Extract observations.
2. Construct a temporary knowledge graph.
3. Detect gaps, conflicts, and cross-domain candidate edges.
4. Generate candidate questions.
5. Rank questions using usefulness, testability, grounding, cross-domain value, and estimated information gain.
6. Investigate selected questions with available evidence tools.
7. Update the graph only with evidence-backed claims; retain hypotheses separately.

## Primary metric
`Knowledge Gain per Question = validated_new_structure / questions_asked`

`validated_new_structure` counts only independently supported new nodes, edges, or resolved contradictions. Rephrasing an existing fact does not count.

## Secondary metrics
- question usefulness
- testability
- factual grounding
- cross-domain value
- redundancy
- human preference
- evidence retrieval success
- cost per validated knowledge unit

## Experimental protocol
1. Freeze the model/version and prompts for a benchmark run.
2. Give baseline and HONET identical source inputs.
3. Generate equal-sized candidate question sets.
4. Blind-rate candidates before showing system identity.
5. Select the same number of questions from each system.
6. Answer/investigate them using the same evidence policy.
7. Measure graph changes and independent evidence.
8. Repeat across multiple domains.
9. Report aggregate results and per-case failures.

## Discovery standard
A result is a **discovery candidate** only when:
- it was not explicitly stated in the source input;
- the relationship is reproducible from the stated evidence or experiment;
- independent evidence supports it;
- the system records the provenance and uncertainty.

The benchmark must never equate model-generated novelty with scientific novelty.

## Stretch target
Find one previously unnoticed, independently verifiable relationship from the benchmark corpus. If found, publish the complete reasoning/evidence trail and attempt independent reproduction.

## Failure is useful
If HONET does not outperform the baseline, retain the result and identify which ranking component failed. The benchmark is designed to falsify the hypothesis, not guarantee it.
