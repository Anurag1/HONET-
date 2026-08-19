import unittest

from leonardo_ai import DiscoveryEngine, Observation


class LeonardoEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = DiscoveryEngine()

    def test_observation_is_recorded_without_explanation(self):
        obs = self.engine.observe([
            Observation("x=1", 1, source="synthetic", confidence=1.0),
            Observation("y=3", 3, source="synthetic", confidence=1.0),
        ])
        self.assertEqual(len(obs), 2)
        self.assertEqual(len(self.engine.hypotheses), 0)

    def test_prediction_and_confirmation(self):
        self.engine.observe([Observation("sample", {"x": 2, "y": 5}, source="synthetic")])
        h = self.engine.hypothesize("y=2x+1", lambda x: 2 * x + 1)
        result = self.engine.experiment(h, 7, 15)
        self.assertEqual(result.status, "CONFIRMED")
        self.assertEqual(result.error, 0.0)

    def test_falsification_creates_explicit_contradiction(self):
        h = self.engine.hypothesize("y=3x", lambda x: 3 * x)
        result = self.engine.experiment(h, 7, 15)
        self.assertEqual(result.status, "FALSIFIED")
        revision = self.engine.revise(h)
        self.assertTrue(revision["revision_required"])
        self.assertEqual(revision["failures"], 1)

    def test_cycle_summary_points_back_to_observation(self):
        h = self.engine.hypothesize("identity", lambda x: x)
        self.engine.experiment(h, 3, 3)
        summary = self.engine.generalize()
        self.assertEqual(summary["next_cycle"], "OBSERVE")
        self.assertEqual(summary["confirmed"], ["identity"])


if __name__ == "__main__":
    unittest.main()
