"""HTTP service for the V7 DesignPreferenceAgent.

The Streamlit application is a presentation layer.  This module exposes the
same model as a small, stateful HTTP service so another UI, an integration, or
an automation agent can call it without duplicating any preference logic.

This MVP deliberately keeps runtime state in memory.  Every mutating request
is also appended to a local JSONL audit file under ``.runtime/`` (or the path
specified by ``DESIGN_AGENT_RUNTIME_DIR``).  The directory is gitignored: it
can contain demo inputs and must not be treated as a training dataset.

Run locally after installing requirements::

    uvicorn agent_api:app --reload --port 8000

The OpenAPI page is then available at http://127.0.0.1:8000/docs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Literal, Mapping, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent import DesignPreferenceAgent


PreferenceDimension = Literal[
    "new_chinese",
    "modern_minimal",
    "warm_wood",
    "warm_lighting",
    "storage_priority",
    "open_space",
    "budget_sensitive",
]
ActionType = Literal["save", "compare", "select", "confirm", "upload", "skip"]


class StrictModel(BaseModel):
    """Reject misspelled input fields instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class QualityInput(StrictModel):
    source_reliability: float = Field(default=0.55, ge=0.0, le=1.0)
    duplicate_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    event_density: float = Field(default=1.0, ge=0.0, le=100.0)


class EvidenceAction(StrictModel):
    type: ActionType
    dwell_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)


class ImageEvidence(StrictModel):
    case_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    tags: List[str] = Field(default_factory=list, max_length=30)

    @field_validator("tags")
    @classmethod
    def tags_are_non_empty(cls, tags: List[str]) -> List[str]:
        if any(not tag.strip() for tag in tags):
            raise ValueError("image tags must not be blank")
        return tags


class EvidenceInput(StrictModel):
    text: List[str] = Field(default_factory=list, max_length=30)
    image_tags: List[str] = Field(default_factory=list, max_length=30)
    images: List[ImageEvidence] = Field(default_factory=list, max_length=20)
    actions: List[EvidenceAction] = Field(default_factory=list, max_length=50)
    quality: QualityInput = Field(default_factory=QualityInput)

    @field_validator("text", "image_tags")
    @classmethod
    def strings_are_non_empty(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence strings must not be blank")
        return values

    @model_validator(mode="after")
    def contains_observable_evidence(self) -> "EvidenceInput":
        if not (self.text or self.image_tags or self.images or self.actions):
            raise ValueError("provide text, image tags/images, or at least one action")
        return self


class ContextInput(StrictModel):
    budget_max: Optional[float] = Field(default=None, gt=0.0, le=100_000_000.0)
    area_sqm: Optional[float] = Field(default=None, gt=0.0, le=100_000.0)
    family_members: Optional[int] = Field(default=None, ge=1, le=50)
    rooms: Optional[int] = Field(default=None, ge=1, le=100)
    stage: Optional[str] = Field(default=None, min_length=1, max_length=64)
    no_structural_change: Optional[bool] = None


class PlanInput(StrictModel):
    plan_id: str = Field(min_length=1, max_length=128)
    name: Optional[str] = Field(default=None, min_length=1, max_length=256)
    estimated_cost: Optional[float] = Field(default=None, ge=0.0, le=100_000_000.0)
    min_area_sqm: Optional[float] = Field(default=None, ge=0.0, le=100_000.0)
    requires_structural_change: bool = False
    features: Dict[PreferenceDimension, float] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def feature_values_are_probabilities(
        cls, features: Dict[PreferenceDimension, float]
    ) -> Dict[PreferenceDimension, float]:
        if any(value < 0.0 or value > 1.0 for value in features.values()):
            raise ValueError("all plan feature values must be between 0 and 1")
        return features


class ResetRequest(StrictModel):
    baseline: Dict[PreferenceDimension, float] = Field(default_factory=dict)

    @field_validator("baseline")
    @classmethod
    def baseline_values_are_probabilities(
        cls, baseline: Dict[PreferenceDimension, float]
    ) -> Dict[PreferenceDimension, float]:
        if any(value < 0.0 or value > 1.0 for value in baseline.values()):
            raise ValueError("all baseline values must be between 0 and 1")
        return baseline


class TurnRequest(StrictModel):
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    timestamp: Optional[datetime] = None
    evidence: EvidenceInput
    context: ContextInput = Field(default_factory=ContextInput)
    confirmed_preferences: List[PreferenceDimension] = Field(default_factory=list, max_length=7)
    plans: List[PlanInput] = Field(default_factory=list, max_length=20)


class AgentRuntime:
    """Owns in-memory project state and an append-only local audit trail."""

    def __init__(self, runtime_dir: Optional[Path] = None) -> None:
        configured_dir = os.getenv("DESIGN_AGENT_RUNTIME_DIR")
        default_runtime_dir = Path(__file__).resolve().parent / ".runtime"
        self.runtime_dir = Path(
            runtime_dir or configured_dir or default_runtime_dir
        ).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.runtime_dir / "agent_audit.jsonl"
        self.agent = DesignPreferenceAgent()
        self._project_ids: set[str] = set()
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = RLock()

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "service": "design-preference-agent-api",
                "projects_in_memory": len(self._project_ids),
                "audit_log": str(self.audit_log_path),
            }

    def reset(self, project_id: str, baseline: Mapping[str, float]) -> Dict[str, Any]:
        with self._lock:
            state = self.agent.reset(project_id, baseline)
            self._project_ids.add(project_id)
            self._history[project_id] = []
            self._append_audit(
                project_id,
                "project_reset",
                {"baseline": dict(baseline)},
            )
            return state

    def process_turn(self, project_id: str, request: TurnRequest) -> Dict[str, Any]:
        with self._lock:
            if project_id not in self._project_ids:
                # A first turn is allowed, but the audit trail makes the implicit reset visible.
                self.reset(project_id, {})

            event = request.model_dump(mode="json", exclude_none=True, exclude={"plans"})
            result = self.agent.respond(
                project_id,
                event,
                [plan.model_dump(mode="json", exclude_none=True) for plan in request.plans],
            )
            history_item = {
                "timestamp": self._now(),
                "session_id": request.session_id,
                "event": event,
                "next_action": result["next_action"],
                "plan_classifications": [
                    {
                        "plan_id": item["plan_id"],
                        "classification": item["classification"],
                        "execution_score_E_ij": item["execution_score_E_ij"],
                    }
                    for item in result["plan_alignment"]
                ],
            }
            self._history.setdefault(project_id, []).append(history_item)
            self._append_audit(project_id, "turn_processed", history_item)
            return result

    def state(self, project_id: str) -> Dict[str, Any]:
        with self._lock:
            self._require_project(project_id)
            return self.agent.get_state(project_id)

    def history(self, project_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            self._require_project(project_id)
            return list(self._history.get(project_id, []))

    def _require_project(self, project_id: str) -> None:
        if project_id not in self._project_ids:
            raise KeyError(project_id)

    def _append_audit(self, project_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        record = {
            "timestamp": self._now(),
            "project_id": project_id,
            "event_type": event_type,
            "payload": payload,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, ensure_ascii=False, default=str))
            audit_file.write("\n")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


def _project_id_or_422(project_id: str) -> str:
    normalized = project_id.strip()
    if not normalized or len(normalized) > 128:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="project_id must contain 1 to 128 non-whitespace characters",
        )
    return normalized


def create_app(runtime: Optional[AgentRuntime] = None) -> FastAPI:
    """Create an injectable app instance; tests use an isolated runtime directory."""

    active_runtime = runtime or AgentRuntime()
    api = FastAPI(
        title="Design Preference Agent API",
        version="0.1.0",
        description="Session-based API for the V7 renovation preference MVP.",
    )

    @api.get("/health")
    def health() -> Dict[str, Any]:
        return active_runtime.health()

    @api.post("/projects/{project_id}/reset")
    def reset_project(project_id: str, request: ResetRequest) -> Dict[str, Any]:
        return active_runtime.reset(_project_id_or_422(project_id), request.baseline)

    @api.post("/projects/{project_id}/turn")
    def process_turn(project_id: str, request: TurnRequest) -> Dict[str, Any]:
        return active_runtime.process_turn(_project_id_or_422(project_id), request)

    @api.get("/projects/{project_id}/state")
    def get_state(project_id: str) -> Dict[str, Any]:
        normalized_id = _project_id_or_422(project_id)
        try:
            return active_runtime.state(normalized_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="project not initialized") from error

    @api.get("/projects/{project_id}/history")
    def get_history(project_id: str) -> Dict[str, Any]:
        normalized_id = _project_id_or_422(project_id)
        try:
            return {"project_id": normalized_id, "turns": active_runtime.history(normalized_id)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="project not initialized") from error

    return api


app = create_app()
