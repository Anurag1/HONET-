from question_engine.commercial import OFFERS, validate_offer_inputs


def test_offers_are_explicit_experiments():
    assert [offer.name for offer in OFFERS] == ["Explorer", "Researcher", "Team Pilot"]
    assert OFFERS[0].price_usd_monthly == 0
    assert OFFERS[1].price_usd_monthly == 29
    assert OFFERS[2].price_usd_monthly == 199


def test_validation_blocks_unsupported_paid_claims():
    gaps = validate_offer_inputs("", [])
    assert "customer_problem_needs_specificity" in gaps
    assert "needs_customer_evidence" in gaps


def test_validation_accepts_specific_problem_with_evidence():
    assert validate_offer_inputs(
        "R&D teams spend too much time finding useful questions in fragmented notes.",
        ["pilot interview"],
    ) == []
