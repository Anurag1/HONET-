from __future__ import annotations

from pydantic import BaseModel, Field
from agents import Agent, Runner


class QuestionCandidate(BaseModel):
    question: str
    novelty: float = Field(ge=0, le=1)
    testability: float = Field(ge=0, le=1)
    cross_domain: float = Field(ge=0, le=1)
    rationale: str


class DiscoveryBatch(BaseModel):
    observations: list[str]
    questions: list[QuestionCandidate]


DISCOVERY_AGENT = Agent(
    name="HONET Question Engine",
    instructions=(
        "Turn observations into high-value research questions. "
        "Separate observations from hypotheses. Prefer cross-domain, testable questions. "
        "Do not claim novelty or truth without evidence. Return structured output."
    ),
    output_type=DiscoveryBatch,
)


def generate_questions(observations: str) -> DiscoveryBatch:
    """Run the question engine through the OpenAI Agents SDK."""
    result = Runner.run_sync(DISCOVERY_AGENT, observations)
    return result.final_output
