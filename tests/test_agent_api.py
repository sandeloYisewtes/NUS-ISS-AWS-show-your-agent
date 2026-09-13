"""Contract tests for the independent HTTP service."""

import json
import shutil
import unittest
from uuid import uuid4
from pathlib import Path

from fastapi.testclient import TestClient

from agent_api import AgentRuntime, create_app


class AgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        # The sandbox does not grant write access to the OS-wide temp directory.
        # Keep disposable audit artifacts under the project's gitignored .runtime.
        runtime_parent = Path(__file__).resolve().parents[1] / ".runtime" / "test-artifacts"
        runtime_parent.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = runtime_parent / f"agent-api-{uuid4().hex}"
        self.runtime_dir.mkdir()
        self.runtime = AgentRuntime(runtime_dir=self.runtime_dir)
        self.client = TestClient(create_app(self.runtime))
        self.project_id = "api_demo_home_001"

    def tearDown(self) -> None:
        shutil.rmtree(self.runtime_dir, ignore_errors=True)

    def _turn_payload(self) -> dict:
        return {
            "session_id": "session-01",
            "timestamp": "2026-09-13T10:00:00+08:00",
            "evidence": {
                "text": ["我喜欢温暖原木和现代简约，收纳必须够。"],
                "images": [{"case_id": "case-1", "tags": ["原木", "暖光"]}],
                "actions": [{"type": "save", "dwell_seconds": 75}],
                "quality": {
                    "source_reliability": 0.9,
                    "duplicate_ratio": 0.05,
                    "event_density": 1.0,
                },
            },
            "context": {
                "budget_max": 180000,
                "area_sqm": 78,
                "no_structural_change": True,
            },
            "plans": [
                {
                    "plan_id": "plan-a",
                    "name": "原木收纳方案",
                    "estimated_cost": 165000,
                    "min_area_sqm": 68,
                    "features": {"warm_wood": 0.9, "storage_priority": 0.9},
                }
            ],
        }

    def test_health_reset_turn_state_history_and_audit_log(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["projects_in_memory"], 0)

        reset = self.client.post(
            f"/projects/{self.project_id}/reset",
            json={"baseline": {"warm_wood": 0.8}},
        )
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["confirmed_dimensions"]["warm_wood"], 0.8)

        turn = self.client.post(f"/projects/{self.project_id}/turn", json=self._turn_payload())
        self.assertEqual(turn.status_code, 200)
        self.assertIn("next_action", turn.json())
        self.assertIn("state", turn.json())

        state = self.client.get(f"/projects/{self.project_id}/state")
        self.assertEqual(state.status_code, 200)
        self.assertEqual(state.json()["sessions_seen"], 1)

        history = self.client.get(f"/projects/{self.project_id}/history")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()["turns"]), 1)
        self.assertEqual(history.json()["turns"][0]["session_id"], "session-01")

        audit_lines = self.runtime.audit_log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(audit_lines), 2)
        self.assertEqual(json.loads(audit_lines[0])["event_type"], "project_reset")
        self.assertEqual(json.loads(audit_lines[1])["event_type"], "turn_processed")

    def test_turn_initializes_project_and_input_is_validated(self) -> None:
        turn = self.client.post(f"/projects/{self.project_id}/turn", json=self._turn_payload())
        self.assertEqual(turn.status_code, 200)

        invalid = self.client.post(
            "/projects/invalid/turn",
            json={
                "evidence": {
                    "text": ["喜欢原木"],
                    "actions": [{"type": "made_up_action", "dwell_seconds": -1}],
                }
            },
        )
        self.assertEqual(invalid.status_code, 422)

        missing = self.client.get("/projects/not-initialized/state")
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
