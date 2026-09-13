"""Regression checks for the synthetic asset/case demonstration bundle."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from run_seed_cases import build_agent_event, load_case, run_case


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "data" / "assets"
CASES_DIR = ROOT / "data" / "cases"


class SyntheticAssetBundleTests(unittest.TestCase):
    def test_catalogue_assets_are_present_and_explicitly_synthetic(self) -> None:
        catalog = json.loads((ASSETS_DIR / "asset_catalog.json").read_text(encoding="utf-8"))
        assets = catalog["assets"]
        self.assertEqual(len(assets), 3)
        for asset in assets:
            self.assertTrue(asset["synthetic_demo"])
            self.assertEqual(asset["source_type"], "ai_generated")
            self.assertEqual(asset["quality_review"]["status"], "human_reviewed")
            self.assertTrue((ROOT / asset["asset_path"]).is_file())
            self.assertGreaterEqual(len(asset["agent_tags"]), 2)
            self.assertLessEqual(len(asset["agent_tags"]), 6)

    def test_cases_keep_hidden_labels_out_of_agent_events(self) -> None:
        case_paths = sorted(CASES_DIR.glob("*.json"))
        self.assertEqual(len(case_paths), 3)
        suggested_splits = set()
        for path in case_paths:
            case = load_case(path)
            self.assertEqual(case["source_type"], "synthetic_demo")
            self.assertTrue(case["synthetic_demo"])
            suggested_splits.add(case["suggested_project_split"])
            self.assertTrue(case["hidden_ground_truth"]["do_not_send_to_agent"])
            self.assertEqual(len(case["sessions"]), 3)
            for session in case["sessions"]:
                event = build_agent_event(session)
                self.assertNotIn("hidden_ground_truth", event)
                self.assertEqual(set(event), {"session_id", "timestamp", "evidence", "context", "confirmed_preferences"})
        self.assertEqual(suggested_splits, {"development", "validation", "test"})

    def test_all_cases_replay_without_using_held_out_labels(self) -> None:
        for path in sorted(CASES_DIR.glob("*.json")):
            report = run_case(load_case(path), case_file=path.name)
            self.assertTrue(report["agent_input_excludes_hidden_ground_truth"])
            self.assertTrue(report["held_out_evaluation"]["read_after_inference_only"])
            self.assertEqual(report["sessions_processed"], 3)
            self.assertIsNotNone(report["top_feasible_plan"])


if __name__ == "__main__":
    unittest.main()
