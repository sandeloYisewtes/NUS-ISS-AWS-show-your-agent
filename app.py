"""现场演示用的 Streamlit 前端。

运行：
    streamlit run app.py

本文件只负责把输入组织为会话级证据包，并把 Agent 的可解释输出翻译为
设计师可读的中文；所有 V7 推断逻辑仍在 ``agent.py`` 中。
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import streamlit as st

from agent import DesignPreferenceAgent


ROOT = Path(__file__).resolve().parent
DEFAULT_PLANS = (ROOT / "data" / "demo_plans.json").read_text(encoding="utf-8")

ACTION_LABELS = {
    "save": "收藏",
    "compare": "对比",
    "select": "明确选择",
    "confirm": "明确确认",
    "upload": "上传案例",
    "skip": "跳过 / 排除",
}

DIMENSION_LABELS = {
    "new_chinese": "新中式 / 东方感",
    "modern_minimal": "现代简约",
    "warm_wood": "温暖原木",
    "warm_lighting": "暖色灯光",
    "storage_priority": "收纳优先",
    "open_space": "通透开放空间",
    "budget_sensitive": "预算敏感度",
}

PLAN_CLASSIFICATION_LABELS = {
    "aligned": "偏好一致：可展示方案",
    "misaligned": "偏好错位：建议先修改方案",
    "needs_owner_confirmation": "需屋主确认：偏好证据仍不足",
    "hard_constraint_failed": "硬约束不通过",
}

NEXT_ACTION_LABELS = {
    "ASK": "追问关键偏好 / 约束",
    "SHOW": "展示对比 Mood Board",
    "CONFIRM": "确认硬约束或补充方案",
    "RECOMMEND": "先修改候选方案",
    "PRESENT": "向屋主展示最佳方案",
    "DEFER": "暂缓结论，补充稳定证据",
}


def _split_tags(raw_tags: str) -> List[str]:
    """把逗号、顿号、分号或换行分隔的图片标签转换成列表。"""
    normalized = raw_tags.replace("，", ",").replace("、", ",").replace("；", ",")
    return [tag.strip() for tag in normalized.replace("\n", ",").split(",") if tag.strip()]


def _parse_plans(raw_plans: str) -> List[Dict[str, Any]]:
    """校验候选方案输入，避免无效 JSON 写入 Agent 状态。"""
    try:
        plans = json.loads(raw_plans)
    except json.JSONDecodeError as error:
        raise ValueError(f"候选方案 JSON 格式错误：{error.msg}（第 {error.lineno} 行）") from error

    if not isinstance(plans, list) or not plans:
        raise ValueError("候选方案必须是至少包含一个方案的 JSON 数组。")

    validated: List[Dict[str, Any]] = []
    for index, plan in enumerate(plans, start=1):
        if not isinstance(plan, dict):
            raise ValueError(f"第 {index} 个候选方案必须是对象。")
        if not plan.get("name"):
            raise ValueError(f"第 {index} 个候选方案缺少 name。")
        features = plan.get("features", {})
        if not isinstance(features, dict):
            raise ValueError(f"第 {index} 个候选方案的 features 必须是对象。")
        validated.append(plan)
    return validated


def _ensure_project(project_id: str) -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = DesignPreferenceAgent()
    if project_id not in st.session_state.get("known_projects", set()):
        st.session_state.agent.reset(project_id)
        st.session_state.known_projects = {
            *st.session_state.get("known_projects", set()),
            project_id,
        }


def _render_preferences(preferences: Sequence[Mapping[str, Any]]) -> None:
    st.subheader("1. 屋主动态偏好后验")
    st.caption("支持度是当前偏好估计；不确定性越低、置信度越高，越适合进入方案决策。")
    rows = []
    for item in preferences:
        label = str(item["label"])
        confidence = float(item["confidence"])
        support = float(item["support"])
        uncertainty = float(item["uncertainty"])
        st.markdown(f"**{label}**　支持度 `{support:.0%}`　置信度 `{confidence:.0%}`")
        st.progress(confidence, text=f"置信度 {confidence:.0%} ｜ 不确定性 {uncertainty:.3f}")
        rows.append(
            {
                "偏好维度": label,
                "支持度": support,
                "不确定性": uncertainty,
                "置信度": confidence,
                "已确认": "是" if item.get("confirmed") else "否",
                "证据": "；".join(item.get("evidence", [])) or "暂无直接证据",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_quality(observation: Mapping[str, Any]) -> None:
    st.subheader("2. 本会话证据质量与注意力")
    quality = observation["quality"]
    quality_col, anomaly_col, noise_col = st.columns(3)
    quality_col.metric("证据质量 q_it", f"{quality['score']:.0%}")
    anomaly_col.metric("异常密度 A_it", f"{quality['anomaly_density']:.0%}")
    noise_col.metric("观测噪声 R_it", f"{observation['measurement_noise_R_it']:.3f}")
    st.caption(
        "异常密度只降低本会话证据的可信度、增大观测噪声；不会直接改变屋主的真实偏好状态。"
    )

    attention = observation.get("attention", [])
    if attention:
        attention_rows = [
            {
                "被模型关注的证据": item["evidence"],
                "关联偏好": DIMENSION_LABELS.get(item["dimension"], item["dimension"]),
                "注意力权重": item["weight"],
            }
            for item in attention
        ]
        st.dataframe(attention_rows, hide_index=True, width="stretch")
    else:
        st.info("未在当前文本或图片标签中识别到预设偏好词。可以补充风格、材质、灯光、收纳或预算描述。")


def _render_plan_alignment(plan_alignment: Sequence[Mapping[str, Any]]) -> None:
    st.subheader("3. 方案执行度与二分类结论")
    if not plan_alignment:
        st.info("尚未输入候选方案，因此当前只完成了屋主偏好估计。")
        return

    summary_rows = [
        {
            "候选方案": item["name"],
            "执行度 E_ij": item["execution_score_E_ij"],
            "硬约束 G_ij": "通过" if item["hard_constraint_ok_G_ij"] else "不通过",
            "结论": PLAN_CLASSIFICATION_LABELS.get(item["classification"], item["classification"]),
        }
        for item in plan_alignment
    ]
    st.dataframe(summary_rows, hide_index=True, width="stretch")

    for item in plan_alignment:
        with st.expander(f"查看「{item['name']}」的错位解释"):
            if item["constraint_reasons"]:
                st.error("；".join(item["constraint_reasons"]))
            else:
                st.success("预算、面积与结构等硬约束均通过。")

            mismatches = item.get("largest_mismatches", [])
            if mismatches:
                mismatch_rows = [
                    {
                        "关键偏好": DIMENSION_LABELS.get(row["dimension"], row["dimension"]),
                        "偏好重要度": row["importance"],
                        "方案体现度": row["plan_embodiment"],
                        "错位程度 Δ_ijk": row["mismatch"],
                    }
                    for row in mismatches
                ]
                st.dataframe(mismatch_rows, hide_index=True, width="stretch")


def _render_next_action(next_action: Mapping[str, Any]) -> None:
    st.subheader("4. Agent 建议的下一步")
    action_type = str(next_action["type"])
    st.markdown(f"### {NEXT_ACTION_LABELS.get(action_type, action_type)}")
    st.write(next_action["reason"])
    payload = next_action.get("payload", {})
    if payload.get("question"):
        st.info(f"建议话术：{payload['question']}")
    elif payload.get("instruction"):
        st.info(f"建议动作：{payload['instruction']}")
    elif payload.get("suggestion"):
        st.info(f"建议动作：{payload['suggestion']}")
    elif payload.get("plan_id"):
        st.info(f"关联方案：{payload['plan_id']}")
    elif payload:
        st.json(payload)


def main() -> None:
    st.set_page_config(page_title="装修偏好共识 Agent", page_icon="🏠", layout="wide")
    st.title("🏠 装修偏好共识 Agent · 现场演示")
    st.write(
        "把屋主的一次浏览/决策会话转成可解释的偏好后验，检查候选方案是否落实偏好，"
        "再建议设计师下一步该追问、展示还是推荐。"
    )
    st.warning("这是可解释 MVP：当前规则与参数用于演示闭环，不应宣称已用真实用户数据训练出业务效果。")

    with st.sidebar:
        st.header("演示控制")
        project_id = st.text_input("项目 / 屋主匿名 ID", value="home_demo_live")
        if st.button("重置当前项目", width="stretch"):
            if "agent" not in st.session_state:
                st.session_state.agent = DesignPreferenceAgent()
            st.session_state.agent.reset(project_id)
            st.session_state.known_projects = {
                *st.session_state.get("known_projects", set()),
                project_id,
            }
            st.session_state.pop("last_result", None)
            st.rerun()
        st.caption("同一项目连续提交多轮会话，才能看到一阶马尔可夫偏好更新的效果。")

    _ensure_project(project_id)

    input_col, plan_col = st.columns((1, 1))
    with input_col:
        st.header("A. 屋主本次会话证据 D_it")
        homeowner_text = st.text_area(
            "屋主搜索词、自然语言描述或语音转写",
            value="我喜欢温暖原木和现代简约，小户型希望通透、收纳多，预算十八万以内。",
            height=120,
            help="例如：我不想要复杂欧式，暖光和原木家具让我感觉放松；收纳必须够。",
        )
        image_tags = st.text_input(
            "图片 / 收藏案例标签（用逗号分隔）",
            value="原木, 现代简约, 收纳",
            help="本 MVP 使用人工或视觉模型提取的标签；未来可接入图片 embedding。",
        )
        selected_actions = st.multiselect(
            "屋主已发生的交互信号",
            options=list(ACTION_LABELS),
            default=["save"],
            format_func=lambda item: ACTION_LABELS[item],
        )
        dwell_seconds = st.number_input("本次总停留时长（秒）", min_value=0, max_value=3600, value=82, step=1)

        with st.expander("证据质量与异常密度输入", expanded=False):
            source_reliability = st.slider("来源可靠度", 0.0, 1.0, 0.88, 0.01)
            duplicate_ratio = st.slider("重复内容比例", 0.0, 1.0, 0.06, 0.01)
            event_density = st.slider("会话事件密度", 0.0, 12.0, 1.2, 0.1)
            st.caption("密度明显高、且重复内容多时，系统会将它视作可能的刷屏/异常证据。")

    with plan_col:
        st.header("B. 外生情境 c_it 与候选方案")
        budget_max = st.number_input("装修预算上限（元）", min_value=0, value=180000, step=10000)
        area_sqm = st.number_input("套内 / 设计面积（㎡）", min_value=0, value=78, step=1)
        family_members = st.number_input("家庭成员数", min_value=1, value=3, step=1)
        no_structural_change = st.checkbox("屋主不接受结构改造", value=True)
        confirmed_dimensions = st.multiselect(
            "本轮已由屋主明确确认的偏好",
            options=list(DIMENSION_LABELS),
            format_func=lambda item: DIMENSION_LABELS[item],
        )
        st.caption("预算、面积、家庭变化等放在外生情境中，不与收藏/点击等原始证据重复计算。")

        with st.expander("编辑候选方案（JSON）", expanded=False):
            plans_text = st.text_area(
                "候选方案列表",
                value=DEFAULT_PLANS,
                height=340,
                help="每个方案至少应包含 name、estimated_cost、min_area_sqm、requires_structural_change 和 features。",
            )
            st.caption("features 的取值范围为 0–1，例如 warm_wood: 0.9 表示方案高度体现温暖原木。")

    submitted = st.button("运行本轮偏好推断与方案评估", type="primary", width="stretch")
    if submitted:
        try:
            plans = _parse_plans(plans_text)
        except ValueError as error:
            st.error(str(error))
            return

        now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
        actions = [
            {"type": action, "dwell_seconds": dwell_seconds / max(len(selected_actions), 1)}
            for action in selected_actions
        ]
        event = {
            "project_id": project_id,
            "session_id": f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": now,
            "evidence": {
                "text": [homeowner_text],
                "image_tags": _split_tags(image_tags),
                "actions": actions,
                "quality": {
                    "source_reliability": source_reliability,
                    "duplicate_ratio": duplicate_ratio,
                    "event_density": event_density,
                },
            },
            "context": {
                "budget_max": budget_max or None,
                "area_sqm": area_sqm or None,
                "family_members": family_members,
                "no_structural_change": no_structural_change,
            },
            "confirmed_preferences": confirmed_dimensions,
        }
        st.session_state.last_result = st.session_state.agent.respond(project_id, event, plans)
        st.session_state.last_event = event

    result = st.session_state.get("last_result")
    if result and result["state"]["project_id"] == project_id:
        st.divider()
        _render_preferences(result["state"]["posterior_preferences"])
        st.divider()
        _render_quality(result["observation"])
        st.divider()
        _render_plan_alignment(result["plan_alignment"])
        st.divider()
        _render_next_action(result["next_action"])
        with st.expander("查看本轮标准化证据包（供评委/技术演示）"):
            st.json(st.session_state.get("last_event", {}))


if __name__ == "__main__":
    main()
