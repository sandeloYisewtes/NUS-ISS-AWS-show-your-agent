"""严格按项目/屋主分组并按时间切分的离线评估工具。

它故意不把同一项目拆进 train 和 test；测试集一旦创建后应冻结，不能用于调参。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _timestamp(record: Mapping[str, Any]) -> datetime:
    return datetime.fromisoformat(str(record["timestamp"]).replace("Z", "+00:00"))


def split_by_project_and_time(
    records: Sequence[Mapping[str, Any]],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Dict[str, List[Dict[str, Any]]]:
    """按项目的最早时间排序，然后整体分入 train/validation/test。

    这是比逐条随机拆分更安全的最小协议。生产场景还应保证 test 位于更晚的
    时间窗口，并固定由版本化 manifest 管理。
    """
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio 与 validation_ratio 必须为正且其和小于 1")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_id = str(record.get("project_id") or record.get("homeowner_id"))
        if not group_id:
            raise ValueError("每条记录必须含 project_id 或 homeowner_id")
        grouped[group_id].append(dict(record))
    ordered_groups = sorted(
        grouped.items(), key=lambda item: min(_timestamp(record) for record in item[1])
    )
    n_groups = len(ordered_groups)
    if n_groups < 3:
        raise ValueError("至少需要 3 个彼此独立的项目/屋主，才能建立 train/validation/test 三个集合")
    # 保留每个集合至少一个完整项目；不能为了机械满足比例而让 test 为空。
    train_cut = min(max(1, int(n_groups * train_ratio)), n_groups - 2)
    validation_cut = min(
        max(train_cut + 1, int(n_groups * (train_ratio + validation_ratio))),
        n_groups - 1,
    )

    result = {"train": [], "validation": [], "test": []}
    for index, (_group_id, rows) in enumerate(ordered_groups):
        split = "train" if index < train_cut else "validation" if index < validation_cut else "test"
        result[split].extend(sorted(rows, key=_timestamp))
    return result


def assert_no_group_leakage(splits: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    seen: Dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            group_id = str(record.get("project_id") or record.get("homeowner_id"))
            previous = seen.setdefault(group_id, split)
            if previous != split:
                raise AssertionError(f"项目/屋主 {group_id} 同时出现在 {previous} 与 {split}")


def metric_notes() -> Dict[str, str]:
    return {
        "preference": "以屋主明确确认、最终选定案例或排序为真值；报告准确率、F1 与校准误差。",
        "alignment": "检验 E_ij 与最终方案接受、修改轮次、满意度的相关性。",
        "policy": "只有记录了动作、候选集合、展示概率和后续反馈时，才能评估/训练 bandit 或 RL。",
        "synthetic": "GAN/模拟数据只能扩充训练集或做鲁棒性测试，绝不能进入验证集或测试集。",
    }
