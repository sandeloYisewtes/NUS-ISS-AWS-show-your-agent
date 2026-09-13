"""V7 装修偏好共识 Agent 的可运行、可解释 MVP。

这个文件故意不训练模型，也不声称已完成完整强化学习：

* 文本、图片标签和屋主已发生行为构成证据 D_it；
* 异常密度只降低本次证据质量 q_it，并通过 R_it 调整贝叶斯更新；
* 隐藏偏好 z_it 以逐维高斯后验的均值/方差近似；
* Agent 只选择下一步辅助动作，不会直接改写屋主真实偏好；
* 动作策略是受硬约束保护的、可解释的 rule-guided contextual bandit 基线。

仅依赖 Python 标准库，便于在比赛现场稳定演示。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import exp, log, sqrt
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EPSILON = 1e-8


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    e_value = exp(value)
    return e_value / (1.0 + e_value)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class Posterior:
    """单一偏好维度 z_ikt 的近似后验 N(mean, variance)。"""

    mean: float = 0.50
    variance: float = 0.25
    evidence: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """同时考虑方向清晰度和不确定性，范围为 0 到 1。"""
        direction = abs(self.mean - 0.5) * 2.0
        certainty = exp(-self.variance / 0.12)
        return _clamp(direction * certainty)


@dataclass
class ProjectState:
    project_id: str
    posterior: Dict[str, Posterior]
    last_timestamp: Optional[datetime] = None
    evidence_history: List[Dict[str, Any]] = field(default_factory=list)
    confirmed_dimensions: Dict[str, float] = field(default_factory=dict)
    history_actions: List[str] = field(default_factory=list)


class DesignPreferenceAgent:
    """把 V7 的核心逻辑压缩为可演示的 Agent API。

    Public API:
        reset(project_id, baseline=None)
        observe(project_id, event)
        evaluate_plans(project_id, plans, context)
        decide(project_id, context, plans)
        respond(project_id, event, plans=None)
    """

    # 维度可以扩展，但第一版只保留设计师能解释的维度。
    DIMENSIONS: Tuple[str, ...] = (
        "new_chinese",
        "modern_minimal",
        "warm_wood",
        "warm_lighting",
        "storage_priority",
        "open_space",
        "budget_sensitive",
    )

    # 短语 -> (偏好维度, 观察值, 人可读证据)。
    LEXICON: Tuple[Tuple[str, str, float, str], ...] = (
        ("新中式", "new_chinese", 0.95, "新中式"),
        ("中式", "new_chinese", 0.78, "中式"),
        ("茶室", "new_chinese", 0.72, "茶室"),
        ("现代简约", "modern_minimal", 0.95, "现代简约"),
        ("极简", "modern_minimal", 0.88, "极简"),
        ("简约", "modern_minimal", 0.72, "简约"),
        ("温暖原木", "warm_wood", 0.98, "温暖原木"),
        ("原木", "warm_wood", 0.88, "原木"),
        ("木饰面", "warm_wood", 0.82, "木饰面"),
        ("暖光", "warm_lighting", 0.93, "暖光"),
        ("暖色灯光", "warm_lighting", 0.95, "暖色灯光"),
        ("收纳", "storage_priority", 0.90, "收纳"),
        ("储物", "storage_priority", 0.85, "储物"),
        ("通透", "open_space", 0.82, "通透"),
        ("开放式", "open_space", 0.80, "开放式"),
        ("小户型", "open_space", 0.66, "小户型的空间利用"),
        ("预算", "budget_sensitive", 0.82, "预算"),
        ("性价比", "budget_sensitive", 0.93, "性价比"),
        ("不超过", "budget_sensitive", 0.88, "预算上限"),
        ("省", "budget_sensitive", 0.72, "节省预算"),
    )

    EXPLICIT_CUES: Tuple[str, ...] = (
        "喜欢",
        "偏好",
        "必须",
        "确认",
        "一定",
        "不想",
        "不要",
        "希望",
    )

    ACTION_WEIGHTS: Mapping[str, float] = {
        "save": 0.75,
        "compare": 0.60,
        "select": 0.95,
        "confirm": 1.00,
        "upload": 0.70,
        "skip": -0.35,
    }

    def __init__(self, prior_mean: float = 0.50, prior_variance: float = 0.25) -> None:
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self._states: Dict[str, ProjectState] = {}

    def reset(
        self,
        project_id: str,
        baseline: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        """初始化一个项目。baseline 是可选的、已确认的先验锚点。"""
        baseline = baseline or {}
        posterior = {
            dimension: Posterior(
                mean=_clamp(float(baseline.get(dimension, self.prior_mean))),
                variance=0.06 if dimension in baseline else self.prior_variance,
            )
            for dimension in self.DIMENSIONS
        }
        self._states[project_id] = ProjectState(
            project_id=project_id,
            posterior=posterior,
            confirmed_dimensions={
                dimension: _clamp(float(value))
                for dimension, value in baseline.items()
                if dimension in self.DIMENSIONS
            },
        )
        return self.get_state(project_id)

    def get_state(self, project_id: str) -> Dict[str, Any]:
        state = self._require_state(project_id)
        return {
            "project_id": project_id,
            "posterior_preferences": self._posterior_payload(state),
            "confirmed_dimensions": dict(state.confirmed_dimensions),
            "history_actions": list(state.history_actions),
            "sessions_seen": len(state.evidence_history),
        }

    def observe(self, project_id: str, event: Mapping[str, Any]) -> Dict[str, Any]:
        """用一个会话级证据包 D_it 更新贝叶斯偏好后验。

        event 的最小形状见 README / data/schema.json。文本、图片标签和行为均在
        evidence 中；预算、面积、家庭阶段等真正外部变化放在 context 中。
        """
        state = self._require_state(project_id)
        timestamp = _parse_time(event.get("timestamp"))
        evidence = dict(event.get("evidence", {}))
        context = dict(event.get("context", {}))

        # 先执行一阶马尔可夫状态预测：上一偏好 + 外部情境，而非将点击重复计入。
        self._predict_transition(state, timestamp, context)

        observation, attention, evidence_notes = self._encode_evidence(evidence)
        quality = self._quality(evidence)
        measurement_noise = 0.14 * (1.0 + quality["anomaly_density"]) / max(
            quality["score"], 0.05
        )

        # 质量 q 只经由 R 调节当次观测权重；不改变真实状态转移噪声。
        updates: List[Dict[str, Any]] = []
        for dimension, observed_value in observation.items():
            prior = state.posterior[dimension]
            kalman_gain = prior.variance / (prior.variance + measurement_noise)
            prior.mean = _clamp(
                prior.mean + kalman_gain * (observed_value - prior.mean)
            )
            prior.variance = max(0.01, (1.0 - kalman_gain) * prior.variance)
            prior.evidence.extend(evidence_notes.get(dimension, []))
            updates.append(
                {
                    "dimension": dimension,
                    "observation": round(observed_value, 3),
                    "kalman_gain": round(kalman_gain, 3),
                }
            )

        # 明确确认是高质量锚点，但仍保留可见证据与小方差。
        for dimension in event.get("confirmed_preferences", []):
            if dimension in state.posterior:
                state.confirmed_dimensions[dimension] = state.posterior[dimension].mean
                state.posterior[dimension].variance = min(
                    state.posterior[dimension].variance, 0.04
                )

        state.last_timestamp = timestamp or state.last_timestamp
        state.evidence_history.append(
            {
                "timestamp": event.get("timestamp"),
                "quality": quality,
                "attention": attention,
                "context": context,
            }
        )
        return {
            "project_id": project_id,
            "quality": quality,
            "measurement_noise_R_it": round(measurement_noise, 3),
            "attention": attention,
            "updates": updates,
            "posterior_preferences": self._posterior_payload(state),
        }

    def evaluate_plans(
        self,
        project_id: str,
        plans: Sequence[Mapping[str, Any]],
        context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """将偏好后验映射为 E_ij、G_ij 和设计师可读解释。"""
        state = self._require_state(project_id)
        context = dict(context or {})
        results: List[Dict[str, Any]] = []
        for plan in plans:
            features = dict(plan.get("features", {}))
            constraint_ok, constraint_reasons = self._hard_constraints(plan, context)
            weighted_mismatch = 0.0
            total_weight = 0.0
            mismatch_details: List[Dict[str, Any]] = []
            for dimension, posterior in state.posterior.items():
                importance = max(0.10, posterior.confidence)
                embodiment = _clamp(float(features.get(dimension, 0.50)))
                # Δ_ijk = p+(1-d) + p- d；这里 p+=mean，p-=1-mean。
                mismatch = posterior.mean * (1.0 - embodiment) + (
                    1.0 - posterior.mean
                ) * embodiment
                weighted_mismatch += importance * mismatch
                total_weight += importance
                if importance >= 0.20:
                    mismatch_details.append(
                        {
                            "dimension": dimension,
                            "mismatch": round(mismatch, 3),
                            "importance": round(importance, 3),
                            "plan_embodiment": round(embodiment, 3),
                        }
                    )
            execution = _clamp(1.0 - weighted_mismatch / max(total_weight, EPSILON))
            coverage_confidence = (
                sum(item["importance"] for item in mismatch_details)
                / max(len(mismatch_details), 1)
            )
            results.append(
                {
                    "plan_id": plan.get("plan_id", "unnamed_plan"),
                    "name": plan.get("name", plan.get("plan_id", "未命名方案")),
                    "execution_score_E_ij": round(execution, 3),
                    "hard_constraint_ok_G_ij": constraint_ok,
                    "coverage_confidence": round(coverage_confidence, 3),
                    "classification": self._classify_plan(
                        execution, constraint_ok, coverage_confidence
                    ),
                    "constraint_reasons": constraint_reasons,
                    "largest_mismatches": sorted(
                        mismatch_details,
                        key=lambda item: item["mismatch"] * item["importance"],
                        reverse=True,
                    )[:3],
                }
            )
        return sorted(
            results,
            key=lambda item: (
                not item["hard_constraint_ok_G_ij"],
                -item["execution_score_E_ij"],
            ),
        )

    def decide(
        self,
        project_id: str,
        context: Optional[Mapping[str, Any]] = None,
        plans: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """在安全动作集合中选择下一步。第一版为规则型 contextual bandit 基线。"""
        state = self._require_state(project_id)
        context = dict(context or {})
        plans = list(plans or [])
        plan_results = self.evaluate_plans(project_id, plans, context) if plans else []
        latest_quality = (
            state.evidence_history[-1]["quality"] if state.evidence_history else None
        )

        # 强质量控制：低质量或异常会话不直接触发设计方案结论。
        if latest_quality and (
            latest_quality["score"] < 0.42
            or latest_quality["anomaly_density"] > 0.80
        ):
            result = {
                "type": "DEFER",
                "reason": "本次证据质量较低或出现异常密度，先补充稳定、可追溯的偏好证据。",
                "payload": {"suggestion": "请屋主选择 2–3 张最喜欢的案例并说明原因。"},
            }
        elif not context.get("budget_max"):
            result = {
                "type": "ASK",
                "reason": "预算是方案硬约束，尚未记录预算上限。",
                "payload": {"dimension": "budget_sensitive", "question": "您的装修总预算上限大约是多少？"},
            }
        else:
            uncertain = self._most_useful_uncertain_dimension(state)
            feasible = [item for item in plan_results if item["hard_constraint_ok_G_ij"]]
            best = feasible[0] if feasible else None
            if uncertain and uncertain["confidence"] < 0.42:
                action_type = "SHOW" if uncertain["dimension"] in {
                    "new_chinese",
                    "modern_minimal",
                    "warm_wood",
                    "warm_lighting",
                } else "ASK"
                result = {
                    "type": action_type,
                    "reason": f"{uncertain['label']}的重要性已出现，但当前置信度不足，先降低不确定性。",
                    "payload": self._clarification_payload(uncertain, action_type),
                }
            elif best and best["execution_score_E_ij"] >= 0.70 and best[
                "coverage_confidence"
            ] >= 0.35:
                result = {
                    "type": "PRESENT",
                    "reason": "候选方案满足硬约束，且已较好落实当前可确认偏好。",
                    "payload": {"plan_id": best["plan_id"], "execution_score": best["execution_score_E_ij"]},
                }
            elif feasible:
                result = {
                    "type": "RECOMMEND",
                    "reason": "存在可行方案，但与高重要度偏好仍有错位；应先修改方案再展示。",
                    "payload": {
                        "plan_id": feasible[0]["plan_id"],
                        "focus": feasible[0]["largest_mismatches"],
                    },
                }
            else:
                result = {
                    "type": "CONFIRM",
                    "reason": "当前候选方案均未通过硬约束；需要先确认约束或补充新方案。",
                    "payload": {"question": "预算、面积和不可变更的结构限制是否已确认？"},
                }
        state.history_actions.append(result["type"])
        return result

    def respond(
        self,
        project_id: str,
        event: Mapping[str, Any],
        plans: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """比赛演示使用的一次调用：observe → plan alignment → next action。"""
        observation = self.observe(project_id, event)
        context = dict(event.get("context", {}))
        alignment = self.evaluate_plans(project_id, plans or [], context)
        action = self.decide(project_id, context, plans or [])
        return {
            "observation": observation,
            "plan_alignment": alignment,
            "next_action": action,
            "state": self.get_state(project_id),
        }

    def _require_state(self, project_id: str) -> ProjectState:
        if project_id not in self._states:
            self.reset(project_id)
        return self._states[project_id]

    def _predict_transition(
        self,
        state: ProjectState,
        timestamp: Optional[datetime],
        context: Mapping[str, Any],
    ) -> None:
        """z_t | z_(t-1), c_t 的简化一阶马尔可夫预测。

        这里的过程噪声 Q_z 表示真实偏好可能随时间和外部情境变化，和 q_it/R_it
        的证据质量机制保持分离。
        """
        delta_days = 7.0
        if timestamp and state.last_timestamp:
            delta_days = max((timestamp - state.last_timestamp).total_seconds() / 86400.0, 0.0)
        persistence = exp(-delta_days / 120.0)
        process_noise = 0.008 + 0.040 * min(delta_days / 120.0, 1.0)

        context_shift = {dimension: 0.0 for dimension in self.DIMENSIONS}
        if context.get("budget_max") and float(context["budget_max"]) <= 180000:
            context_shift["budget_sensitive"] = 0.07
        if context.get("area_sqm") and float(context["area_sqm"]) <= 85:
            context_shift["storage_priority"] = 0.04
            context_shift["open_space"] = 0.04
        if context.get("family_members") and int(context["family_members"]) >= 4:
            context_shift["storage_priority"] = 0.05

        for dimension, posterior in state.posterior.items():
            posterior.mean = _clamp(
                self.prior_mean
                + persistence * (posterior.mean - self.prior_mean)
                + context_shift[dimension]
            )
            posterior.variance = min(0.35, posterior.variance + process_noise)

    def _encode_evidence(
        self, evidence: Mapping[str, Any]
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, List[str]]]:
        texts = [str(item) for item in evidence.get("text", [])]
        texts.extend(str(item) for item in evidence.get("image_tags", []))
        for image in evidence.get("images", []):
            texts.extend(str(tag) for tag in image.get("tags", []))
        corpus = " ".join(texts).lower()
        actions = evidence.get("actions", [])
        if isinstance(actions, Mapping):
            actions = [actions]
        action_strength = self._action_strength(actions)
        dwell_seconds = sum(
            max(0.0, float(action.get("dwell_seconds", 0) or 0))
            for action in actions
            if isinstance(action, Mapping)
        )
        dwell_factor = min(0.30, log(1.0 + dwell_seconds) / 15.0)
        explicit_factor = 0.30 if any(cue in corpus for cue in self.EXPLICIT_CUES) else 0.0

        raw_items: List[Tuple[str, float, str]] = []
        notes: Dict[str, List[str]] = {dimension: [] for dimension in self.DIMENSIONS}
        for phrase, dimension, score, label in self.LEXICON:
            if phrase.lower() in corpus:
                # 只将直接否定该短语的简单形式视为反向信号，避免复杂 NLP 假装准确。
                is_negated = any(
                    marker + phrase in corpus for marker in ("不喜欢", "不要", "不想要")
                )
                value = 1.0 - score if is_negated else score
                raw_weight = max(0.05, score + explicit_factor + action_strength + dwell_factor)
                raw_items.append((dimension, raw_weight, label))
                notes[dimension].append(("排除 " if is_negated else "提及 ") + label)

        grouped: Dict[str, List[Tuple[float, float, str]]] = {
            dimension: [] for dimension in self.DIMENSIONS
        }
        for phrase, dimension, score, _label in self.LEXICON:
            if phrase.lower() in corpus:
                is_negated = any(
                    marker + phrase in corpus for marker in ("不喜欢", "不要", "不想要")
                )
                value = 1.0 - score if is_negated else score
                weight = max(0.05, score + explicit_factor + action_strength + dwell_factor)
                grouped[dimension].append((value, weight, phrase))

        observation: Dict[str, float] = {}
        for dimension, items in grouped.items():
            if items:
                weight_sum = sum(weight for _value, weight, _phrase in items)
                observation[dimension] = _clamp(
                    sum(value * weight for value, weight, _phrase in items) / weight_sum
                )

        total_attention = sum(weight for _dimension, weight, _label in raw_items)
        attention = [
            {
                "evidence": label,
                "dimension": dimension,
                "weight": round(weight / max(total_attention, EPSILON), 3),
            }
            for dimension, weight, label in raw_items
        ]
        return observation, attention, {k: v for k, v in notes.items() if v}

    def _action_strength(self, actions: Iterable[Any]) -> float:
        weights: List[float] = []
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            action_type = str(action.get("type", "")).lower()
            weights.append(self.ACTION_WEIGHTS.get(action_type, 0.0))
        return _clamp(sum(weights) / max(len(weights), 1), -0.35, 1.0)

    def _quality(self, evidence: Mapping[str, Any]) -> Dict[str, float]:
        quality = dict(evidence.get("quality", {}))
        source = _clamp(float(quality.get("source_reliability", 0.55)))
        duplicate = _clamp(float(quality.get("duplicate_ratio", 0.0)))
        density = max(0.0, float(quality.get("event_density", 1.0)))
        # 演示版异常密度：高于正常会话密度 4 的部分视为可疑，范围压缩到 [0,1]。
        anomaly_density = _clamp((density - 4.0) / 8.0 + duplicate * 0.35)
        texts = " ".join(str(item) for item in evidence.get("text", []))
        explicit = 1.0 if any(cue in texts for cue in self.EXPLICIT_CUES) else 0.0
        actions = evidence.get("actions", [])
        if isinstance(actions, Mapping):
            actions = [actions]
        saved_or_confirmed = any(
            isinstance(action, Mapping)
            and str(action.get("type", "")).lower() in {"save", "select", "confirm", "upload"}
            for action in actions
        )
        dwell_seconds = sum(
            max(0.0, float(action.get("dwell_seconds", 0) or 0))
            for action in actions
            if isinstance(action, Mapping)
        )
        score = _sigmoid(
            -0.55
            + 0.80 * source
            + 0.75 * explicit
            + 0.55 * float(saved_or_confirmed)
            + 0.15 * min(log(1.0 + dwell_seconds), 5.0)
            - 1.35 * duplicate
            - 1.15 * anomaly_density
        )
        return {
            "score": round(_clamp(score), 3),
            "source_reliability": round(source, 3),
            "duplicate_ratio": round(duplicate, 3),
            "event_density": round(density, 3),
            "anomaly_density": round(anomaly_density, 3),
        }

    def _hard_constraints(
        self, plan: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        budget_max = context.get("budget_max")
        plan_cost = plan.get("estimated_cost")
        if budget_max is not None and plan_cost is not None and float(plan_cost) > float(budget_max):
            reasons.append("预计成本超过屋主预算上限")
        area = context.get("area_sqm")
        min_area = plan.get("min_area_sqm")
        if area is not None and min_area is not None and float(min_area) > float(area):
            reasons.append("方案所需面积高于当前户型面积")
        if context.get("no_structural_change") and plan.get("requires_structural_change"):
            reasons.append("方案需要结构改造，但屋主要求不改结构")
        return not reasons, reasons

    def _posterior_payload(self, state: ProjectState) -> List[Dict[str, Any]]:
        payload = []
        for dimension, posterior in state.posterior.items():
            payload.append(
                {
                    "dimension": dimension,
                    "label": self._label(dimension),
                    "support": round(posterior.mean, 3),
                    "uncertainty": round(sqrt(posterior.variance), 3),
                    "confidence": round(posterior.confidence, 3),
                    "evidence": posterior.evidence[-5:],
                    "confirmed": dimension in state.confirmed_dimensions,
                }
            )
        return sorted(payload, key=lambda item: item["confidence"], reverse=True)

    def _most_useful_uncertain_dimension(self, state: ProjectState) -> Optional[Dict[str, Any]]:
        candidates = []
        for dimension, posterior in state.posterior.items():
            # 有一定方向性，但置信度不足时最值得澄清。
            relevance = abs(posterior.mean - 0.5) * 2.0
            candidates.append(
                {
                    "dimension": dimension,
                    "label": self._label(dimension),
                    "confidence": posterior.confidence,
                    "relevance": relevance,
                }
            )
        return max(candidates, key=lambda item: item["relevance"] * (1.0 - item["confidence"]))

    def _clarification_payload(self, item: Mapping[str, Any], action_type: str) -> Dict[str, Any]:
        if action_type == "SHOW":
            return {
                "dimension": item["dimension"],
                "instruction": f"展示两组在「{item['label']}」上差异明显的 mood board，并请屋主二选一。",
            }
        return {
            "dimension": item["dimension"],
            "question": f"「{item['label']}」对您是必须、偏好还是可有可无？",
        }

    @staticmethod
    def _classify_plan(execution: float, constraint_ok: bool, confidence: float) -> str:
        if not constraint_ok:
            return "hard_constraint_failed"
        if confidence < 0.30:
            return "needs_owner_confirmation"
        return "aligned" if execution >= 0.70 else "misaligned"

    @staticmethod
    def _label(dimension: str) -> str:
        labels = {
            "new_chinese": "新中式 / 东方感",
            "modern_minimal": "现代简约",
            "warm_wood": "温暖原木",
            "warm_lighting": "暖色灯光",
            "storage_priority": "收纳优先",
            "open_space": "通透开放空间",
            "budget_sensitive": "预算敏感度",
        }
        return labels.get(dimension, dimension)
