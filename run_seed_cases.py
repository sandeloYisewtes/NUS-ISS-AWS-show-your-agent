"""回放合成案例，并将隐藏评估标签严格隔离在 Agent 输入之外。

这不是训练脚本：它用三组 ``synthetic_demo`` 会话验证素材库、标签投影和
V7 Agent 的端到端流程。真实数据接入后，应以项目为单位独立切分训练、验证、
测试集，且测试标签不可参与阈值或词典调整。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from agent import DesignPreferenceAgent


ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "data" / "cases"
AGENT_EVENT_KEYS = (
    "session_id",
    "timestamp",
    "evidence",
    "context",
    "confirmed_preferences",
)


def _configure_utf8_stdout() -> None:
    """Keep Chinese JSON readable when the script is run from older Windows consoles."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_case(path: Path) -> dict[str, Any]:
    """读取一个合成案例并做最小的来源安全检查。"""
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("source_type") != "synthetic_demo":
        raise ValueError(f"{path.name}: expected source_type=synthetic_demo")
    if case.get("synthetic_demo") is not True:
        raise ValueError(f"{path.name}: synthetic_demo marker is required")
    if not case.get("project_id") or not case.get("sessions"):
        raise ValueError(f"{path.name}: project_id and sessions are required")
    hidden = case.get("hidden_ground_truth", {})
    if hidden.get("do_not_send_to_agent") is not True:
        raise ValueError(f"{path.name}: hidden_ground_truth must be explicitly isolated")
    return case


def build_agent_event(session: Mapping[str, Any]) -> dict[str, Any]:
    """用白名单构建单次 Agent 输入，永不携带案例级隐藏标签。

    ``hidden_ground_truth`` 位于案例根对象中；本函数只接受一个 session，并仅复制
    V7 Agent 的公开证据字段。因此，即使案例文件含有人工评估答案，也不会进入
    ``observe`` 或 ``respond``。
    """
    missing = {"session_id", "timestamp", "evidence", "context"} - set(session)
    if missing:
        raise ValueError(f"session missing required fields: {sorted(missing)}")
    return {
        key: copy.deepcopy(session[key])
        for key in AGENT_EVENT_KEYS
        if key in session
    }


def _first_feasible(plan_alignment: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next(
        (
            plan
            for plan in plan_alignment
            if plan.get("hard_constraint_ok_G_ij") is True
        ),
        None,
    )


def run_case(case: Mapping[str, Any], *, case_file: str = "<in-memory>") -> dict[str, Any]:
    """独立回放一个项目；推断结束后才读取 held-out 评估标签。"""
    project_id = str(case["project_id"])
    plans = copy.deepcopy(case.get("candidate_plans", []))
    agent = DesignPreferenceAgent()
    agent.reset(project_id)

    latest_result: dict[str, Any] | None = None
    for session in case["sessions"]:
        # 这是输入边界：不传 case 根对象，隐藏标签也不会和 event 合并。
        latest_result = agent.respond(
            project_id,
            build_agent_event(session),
            plans=plans,
        )

    if latest_result is None:  # load_case 已检查，保留防御式分支。
        raise ValueError(f"{case_file}: no sessions to replay")

    best = _first_feasible(latest_result["plan_alignment"])

    # 只在完整推断之后才读取下面的人工预设答案，用于报告而非模型输入。
    hidden = case["hidden_ground_truth"]
    expected_plan_id = hidden.get("preferred_plan_id")
    expected_classification = hidden.get("expected_final_plan_classification")
    expected_action = hidden.get("expected_next_action_after_session_1")

    return {
        "case_file": case_file,
        "project_id": project_id,
        "title": case.get("title"),
        "source_type": case.get("source_type"),
        "suggested_project_split": case.get("suggested_project_split"),
        "sessions_processed": len(case["sessions"]),
        "agent_input_excludes_hidden_ground_truth": True,
        "final_next_action": latest_result["next_action"]["type"],
        "top_feasible_plan": (
            {
                "plan_id": best["plan_id"],
                "name": best["name"],
                "execution_score_E_ij": best["execution_score_E_ij"],
                "hard_constraint_ok_G_ij": best["hard_constraint_ok_G_ij"],
                "classification": best["classification"],
            }
            if best
            else None
        ),
        "held_out_evaluation": {
            "read_after_inference_only": True,
            "expected_preferred_plan_id": expected_plan_id,
            "agent_top_feasible_plan_id": best["plan_id"] if best else None,
            "preferred_plan_match": bool(best and best["plan_id"] == expected_plan_id),
            "expected_final_plan_classification": expected_classification,
            "agent_top_plan_classification": best["classification"] if best else None,
            "expected_action_after_session_1": expected_action,
            "note": "A mismatch is a diagnostic signal, not a label to feed back into the current inference run.",
        },
    }


def main() -> int:
    _configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Replay synthetic Design Preference Agent cases.")
    parser.add_argument(
        "--case",
        dest="case_name",
        help="Optional case file name under data/cases, for example home_001_warm_modern_wood.json.",
    )
    args = parser.parse_args()

    paths = (
        [CASES_DIR / args.case_name]
        if args.case_name
        else sorted(CASES_DIR.glob("*.json"))
    )
    if not paths:
        raise SystemExit("No case JSON files found.")
    if any(not path.is_file() for path in paths):
        raise SystemExit("Requested case file does not exist under data/cases.")

    reports = [run_case(load_case(path), case_file=path.name) for path in paths]
    print(
        json.dumps(
            {
                "run_type": "synthetic_demo_workflow_validation",
                "training_performed": False,
                "warning": "These are synthetic demonstrations, not real homeowner data or real-world performance metrics.",
                "cases": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
