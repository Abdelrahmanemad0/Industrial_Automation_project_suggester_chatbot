"""Deterministic, dependency-free retrieval/scoring layer.

This is the "ground truth" the LLM is anchored to. It never hallucinates -
every candidate it returns is a real, curated project from data/projects.json.
The LLM's job (backend/llm.py) is only to personalize the phrasing and
ranking rationale on top of this grounded shortlist; if the LLM is
unavailable, these scored candidates alone are enough to answer the user
(backend/engine.py falls back to templated text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from backend.catalog import Catalog, get_catalog

SKILL_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "i", "want", "like", "interested", "into", "project", "projects",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


@dataclass
class ScoredProject:
    project: dict
    total_score: float
    hardware_score: float
    skill_score: float
    time_score: float
    interest_score: float
    have_required: List[str]
    missing_required: List[str]
    have_optional: List[str]


def _skill_score(user_skill: str, project_skill: str) -> float:
    if user_skill not in SKILL_RANK or project_skill not in SKILL_RANK:
        return 0.5
    diff = abs(SKILL_RANK[user_skill] - SKILL_RANK[project_skill])
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.35
    return 0.0


def _time_score(user_hours: float, project_hours: float) -> float:
    if user_hours <= 0:
        return 0.5
    if project_hours <= user_hours:
        return 1.0
    overrun = (project_hours - user_hours) / user_hours
    return max(0.0, 1.0 - overrun)


def _hardware_score(available: set[str], required: List[str], optional: List[str]):
    have_required = [c for c in required if c in available]
    missing_required = [c for c in required if c not in available]
    have_optional = [c for c in optional if c in available]

    req_coverage = len(have_required) / len(required) if required else 1.0
    if optional:
        opt_coverage = len(have_optional) / len(optional)
        score = 0.75 * req_coverage + 0.25 * opt_coverage
    else:
        # No optional parts exist for this project - don't let an empty
        # optional list cap the score below full required coverage.
        score = req_coverage
    return score, have_required, missing_required, have_optional


def _interest_score(interest_text: str, project: dict) -> float:
    interest_tokens = _tokenize(interest_text)
    if not interest_tokens:
        return 0.5  # neutral - no signal either way
    project_tokens = _tokenize(project["category"]) | _tokenize(project["title"]) | _tokenize(
        project["description"]
    )
    if not project_tokens:
        return 0.0
    overlap = interest_tokens & project_tokens
    return min(1.0, len(overlap) / max(2, len(interest_tokens)))


WEIGHTS = {
    "hardware": 0.40,
    "skill": 0.25,
    "time": 0.15,
    "interest": 0.20,
}


def score_projects(
    skill_level: str,
    available_components: List[str],
    time_budget_hours: float,
    interests: str,
    catalog: Catalog | None = None,
) -> List[ScoredProject]:
    catalog = catalog or get_catalog()
    available = set(available_components)

    results: List[ScoredProject] = []
    for project in catalog.projects:
        hw_score, have_req, missing_req, have_opt = _hardware_score(
            available, project["required_components"], project["optional_components"]
        )
        sk_score = _skill_score(skill_level, project["skill_level"])
        tm_score = _time_score(time_budget_hours, project["time_hours"])
        it_score = _interest_score(interests, project)

        total = (
            WEIGHTS["hardware"] * hw_score
            + WEIGHTS["skill"] * sk_score
            + WEIGHTS["time"] * tm_score
            + WEIGHTS["interest"] * it_score
        )

        results.append(
            ScoredProject(
                project=project,
                total_score=round(total, 4),
                hardware_score=round(hw_score, 4),
                skill_score=round(sk_score, 4),
                time_score=round(tm_score, 4),
                interest_score=round(it_score, 4),
                have_required=have_req,
                missing_required=missing_req,
                have_optional=have_opt,
            )
        )

    results.sort(key=lambda r: r.total_score, reverse=True)
    return results


def top_candidates(
    skill_level: str,
    available_components: List[str],
    time_budget_hours: float,
    interests: str,
    k: int = 4,
    catalog: Catalog | None = None,
) -> List[ScoredProject]:
    return score_projects(
        skill_level, available_components, time_budget_hours, interests, catalog
    )[:k]
