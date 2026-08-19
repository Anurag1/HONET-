from __future__ import annotations

import base64
from pathlib import Path

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


def generate_questions_from_image(image_path: str) -> DiscoveryBatch:
    """Send a local notebook image through the Responses-format multimodal input."""
    data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    input_items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Extract the important observations from this notebook image, "
                        "then generate high-value cross-domain research questions. "
                        "Do not treat uncertain handwriting as fact."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{data}",
                },
            ],
        }
    ]
    result = Runner.run_sync(DISCOVERY_AGENT, input_items)
    return result.final_output
