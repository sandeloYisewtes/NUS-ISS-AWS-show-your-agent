from __future__ import annotations

import unittest

from agent import DesignPreferenceAgent
from evaluate import assert_no_group_leakage, split_by_project_and_time


class DesignPreferenceAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DesignPreferenceAgent()
        self.project_id = "test_home"
        self.agent.reset(self.project_id)

    def test_high_quality_evidence_moves_relevant_posterior(self) -> None:
        result = self.agent.observe(
            self.project_id,
            {
                "timestamp": "2026-09-01T10:00:00+08:00",
                "evidence": {
                    "text": ["我喜欢温暖原木和暖光，收纳必须充足"],
                    "actions": [{"type": "save", "dwell_seconds": 90}],
                    "quality": {"source_reliability": 0.9, "duplicate_ratio": 0.0, "event_density": 1.0},
                },
                "context": {"budget_max": 180000, "area_sqm": 78},
            },
        )
        preferences = {item["dimension"]: item for item in result["posterior_preferences"]}
        self.assertGreater(preferences["warm_wood"]["support"], 0.50)
        self.assertGreater(preferences["warm_lighting"]["support"], 0.50)
        self.assertGreater(result["quality"]["score"], 0.50)

    def test_budget_constraint_blocks_plan(self) -> None:
        self.agent.observe(
            self.project_id,
            {
                "timestamp": "2026-09-01T10:00:00+08:00",
                "evidence": {
                    "text": ["预算不超过十八万，喜欢原木"],
                    "actions": [{"type": "confirm", "dwell_seconds": 60}],
                    "quality": {"source_reliability": 0.9, "duplicate_ratio": 0.0, "event_density": 1.0},
                },
                "context": {"budget_max": 180000, "area_sqm": 78},
            },
        )
        scored = self.agent.evaluate_plans(
            self.project_id,
            [{"plan_id": "too_expensive", "estimated_cost": 250000, "features": {"warm_wood": 0.9}}],
            {"budget_max": 180000, "area_sqm": 78},
        )
        self.assertFalse(scored[0]["hard_constraint_ok_G_ij"])
        self.assertEqual(scored[0]["classification"], "hard_constraint_failed")

    def test_grouped_split_has_no_project_leakage(self) -> None:
        records = [
            {"project_id": f"p{project}", "timestamp": f"2026-09-0{project}T10:00:00+08:00", "x": row}
            for project in range(1, 8)
            for row in range(2)
        ]
        splits = split_by_project_and_time(records)
        assert_no_group_leakage(splits)
        self.assertTrue(all(splits[name] for name in ("train", "validation", "test")))
        memberships = {
            item["project_id"]: split
            for split, rows in splits.items()
            for item in rows
        }
        self.assertEqual(len(memberships), 7)


if __name__ == "__main__":
    unittest.main()
