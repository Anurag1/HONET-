import unittest

from examples.gbrain_honet_benchmark import baseline, honet_discovery


class GBrainHONETBenchmarkTest(unittest.TestCase):
    def test_baseline_retrieves_evidence(self):
        result = baseline("which company has lower latency and why?")
        self.assertGreaterEqual(len(result["retrieved"]), 2)
        self.assertIn("latency_ms", result["answer"])

    def test_discovery_surfaces_falsifiable_question(self):
        result = honet_discovery("which company has lower latency and why?")
        self.assertTrue(any("falsify" in q.lower() for q in result["questions"]))
        self.assertEqual(result["experiment"].status, "FALSIFIED")
        self.assertTrue(result["revision"]["revision_required"])


if __name__ == "__main__":
    unittest.main()
