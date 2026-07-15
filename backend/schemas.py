from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SkillLevel = Literal["beginner", "intermediate", "advanced"]


class SuggestRequest(BaseModel):
    skill_level: SkillLevel = "beginner"
    available_hardware: List[str] = Field(default_factory=list)
    interests: str = ""
    time_budget_hours: float = 8.0

    @field_validator("time_budget_hours")
    @classmethod
    def clamp_time(cls, v: float) -> float:
        # Human-proofing: never let a bad/negative/absurd input reach the engine.
        if v is None or v <= 0:
            return 4.0
        return min(v, 200.0)

    @field_validator("available_hardware")
    @classmethod
    def cap_hardware_list(cls, v: List[str]) -> List[str]:
        return v[:50]

    @field_validator("interests")
    @classmethod
    def cap_interests(cls, v: str) -> str:
        return (v or "")[:500]


class Suggestion(BaseModel):
    project_id: str
    title: str
    category: str
    skill_level: SkillLevel
    time_hours: float
    match_score: float
    have_components: List[str]
    missing_components: List[str]
    optional_components_owned: List[str]
    description: str
    learning_outcomes: List[str]
    rationale: str  # personalized "why this fits you" text
    source: Literal["fine_tuned_llm", "template_fallback"]


class SuggestResponse(BaseModel):
    suggestions: List[Suggestion]
    engine_mode: Literal["fine_tuned_llm", "template_fallback"]
    notes: Optional[str] = None
