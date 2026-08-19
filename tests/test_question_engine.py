from question_engine.engine import CandidateQuestion, information_gain, rank_questions


def test_information_gain_increases_with_new_structure():
    assert information_gain(2, 3, 0) > information_gain(1, 0, 0)


def test_question_ranking_prefers_high_value_question():
    low = CandidateQuestion(
        "What is Pascal's triangle?", 0.2, 0.9, 0.1, 0.2
    )
    high = CandidateQuestion(
        "Can graph structure unify Pascal and interpolation patterns?",
        0.9,
        0.9,
        0.95,
        0.8,
    )
    ranked = rank_questions([low, high])
    assert ranked[0] == high
    assert ranked[0].score > ranked[1].score
