from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommercialOffer:
    name: str
    description: str
    price_usd_monthly: int | None
    target_customer: str


OFFERS = (
    CommercialOffer(
        name="Explorer",
        description="Limited question-discovery analyses for individual exploration.",
        price_usd_monthly=0,
        target_customer="Students, independent researchers, and curious builders",
    ),
    CommercialOffer(
        name="Researcher",
        description="Higher-volume document and notebook analysis with benchmark reports.",
        price_usd_monthly=29,
        target_customer="Researchers and technical professionals",
    ),
    CommercialOffer(
        name="Team Pilot",
        description="Custom knowledge-discovery workflow for a small R&D team.",
        price_usd_monthly=199,
        target_customer="Small R&D and engineering teams",
    ),
)


def validate_offer_inputs(customer_problem: str, evidence: list[str]) -> list[str]:
    """Return validation gaps before presenting a paid offer."""
    gaps: list[str] = []
    if len(customer_problem.strip()) < 20:
        gaps.append("customer_problem_needs_specificity")
    if not evidence:
        gaps.append("needs_customer_evidence")
    return gaps
