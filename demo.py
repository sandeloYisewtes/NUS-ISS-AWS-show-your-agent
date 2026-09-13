"""回放三轮屋主会话，展示 V7 Agent 的闭环。"""

from __future__ import annotations

import json
from pathlib import Path

from agent import DesignPreferenceAgent


ROOT = Path(__file__).resolve().parent


def main() -> None:
    events = [
        json.loads(line)
        for line in (ROOT / "data" / "demo_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plans = json.loads((ROOT / "data" / "demo_plans.json").read_text(encoding="utf-8"))
    project_id = "home_demo_001"
    agent = DesignPreferenceAgent()
    agent.reset(project_id)

    for round_no, event in enumerate(
        [item for item in events if item["project_id"] == project_id], start=1
    ):
        result = agent.respond(project_id, event, plans)
        print(f"\n{'=' * 20} 第 {round_no} 轮会话 {'=' * 20}")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
