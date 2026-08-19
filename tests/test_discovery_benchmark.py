import unittest

from benchmarks.discovery_benchmark import demo_cases, score_cases


class DiscoveryBenchmarkTests(unittest.TestCase):
    def test_honet_has_nonzero_recall(self):
        _, honet = score_cases(demo_cases())
        self.assertGreater(honet.recall, 0.0)

    def test_honet_detects_the_synthetic_contradiction(self):
        _, honet = score_cases(demo_cases())
        self.assertEqual(honet.contradiction_detection, 1)

    def test_metrics_are_bounded(self):
        for score in score_cases(demo_cases()):
            for metric in (score.precision, score.recall, score.novelty_rate, score.contradiction_recall):
                self.assertGreaterEqual(metric, 0.0)
                self.assertLessEqual(metric, 1.0)


if __name__ == "__main__":
    unittest.main()
