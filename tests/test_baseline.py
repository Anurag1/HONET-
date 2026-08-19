from question_engine.baseline import compare
from question_engine.engine import CandidateQuestion


def test_compare_ranks_generated_questions():
    low = CandidateQuestion("Define binary", 0.1, 0.9, 0.1, 0.1)
    high = CandidateQuestion("Test whether number representations reveal a common graph structure", 0.9, 0.9, 0.9, 0.9)
    result = compare([low], [low, high])
    assert result.ranked[0] == high
