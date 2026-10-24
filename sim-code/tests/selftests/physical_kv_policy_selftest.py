"""Small causal tests for the physical-progress KV replay."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "serving_study"))
from kv_progress_replay import replay
from evaluate_progress import evaluate
from physical_kv_policy import NextCallPredictor, ProgressUpdate


class PhysicalKVPolicyTest(unittest.TestCase):
    def test_progress_update_changes_next_call_order(self):
        predictor = NextCallPredictor(default_gap_s=10)
        a, b = ("a", "planner_shadow"), ("b", "planner_shadow")
        predictor.observe_call(a, 0)
        predictor.observe_call(b, 0)
        predictor.observe_progress(ProgressUpdate(1, "a", "planner_shadow", "navigation",
                                                  remaining_path_m=1, linear_speed_mps=1))
        predictor.observe_progress(ProgressUpdate(1, "b", "planner_shadow", "navigation",
                                                  remaining_path_m=9, linear_speed_mps=1))
        self.assertEqual(predictor.estimate(a, 1, True)[0], 1)
        self.assertEqual(predictor.estimate(b, 1, True)[0], 9)
        self.assertEqual(predictor.estimate(a, 1, False)[0], 9)

    def test_update_is_consumed_at_its_time_and_eviction_changes(self):
        requests = [
            {"time_s": -20, "look": [], "all": [], "session": ("a", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": -1, "look": [], "all": [], "session": ("b", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": 0, "look": [], "all": ["a"], "session": ("a", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": 0.5, "look": [], "all": ["b"], "session": ("b", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": 2, "look": [], "all": ["c"], "session": ("c", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": 3, "look": ["a"], "all": ["a"], "session": ("a", "planner_shadow"), "role": "planner_shadow"},
        ]
        updates = [
            ProgressUpdate(1, "a", "planner_shadow", "navigation", remaining_path_m=1, linear_speed_mps=1),
            ProgressUpdate(1, "b", "planner_shadow", "navigation", remaining_path_m=100, linear_speed_mps=1),
        ]
        self.assertEqual(replay(requests, 2, "progress", updates)["total_misses"], 0)
        self.assertEqual(replay(requests, 2, "lru", updates)["total_misses"], 1)
        self.assertEqual(replay(requests, 2, "interval", updates)["total_misses"], 1)

    def test_future_update_cannot_affect_earlier_eviction(self):
        requests = [
            {"time_s": 0, "look": [], "all": ["a"], "session": ("a", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": 1, "look": [], "all": ["b"], "session": ("b", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": 2, "look": [], "all": ["c"], "session": ("c", "planner_shadow"), "role": "planner_shadow"},
            {"time_s": 3, "look": ["a"], "all": ["a"], "session": ("a", "planner_shadow"), "role": "planner_shadow"},
        ]
        future = [ProgressUpdate(2.5, "a", "planner_shadow", "navigation",
                                 remaining_path_m=0, linear_speed_mps=1)]
        self.assertEqual(replay(requests, 2, "progress", future)["total_misses"],
                         replay(requests, 2, "interval")["total_misses"])

    def test_done_and_validation(self):
        predictor = NextCallPredictor()
        predictor.observe_progress(ProgressUpdate(1, "a", "planner_shadow", "done"))
        self.assertTrue(predictor.estimate(("a", "planner_shadow"), 2, True)[2])
        with self.assertRaises(ValueError):
            ProgressUpdate.from_record({"time_s": 1, "episode_id": "a", "role": "planner_shadow",
                                        "phase": "navigation", "remaining_path_m": -1})

    def test_prediction_evaluation_uses_later_calls_only_as_labels(self):
        calls = [{"time_s": time, "session": (episode, "planner_shadow")}
                 for time, episode in [(0, "a"), (0, "b"), (3, "a"), (8, "b")]]
        updates = [
            ProgressUpdate(1, "a", "planner_shadow", "navigation", remaining_path_m=2, linear_speed_mps=1),
            ProgressUpdate(1, "b", "planner_shadow", "navigation", remaining_path_m=7, linear_speed_mps=1),
        ]
        result = evaluate(calls, updates)
        self.assertEqual(result["mean_absolute_error_s"]["progress"], 0)
        self.assertEqual(result["pairwise_accuracy"]["progress"], 1)
        self.assertEqual(result["pairwise_accuracy"]["interval"], 0)


if __name__ == "__main__":
    unittest.main()
