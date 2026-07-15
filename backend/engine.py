"""Orchestrates retrieval + (optional) fine-tuned LLM personalization,
with strict validation and a deterministic fallback.

Design goal ("human proof"): the API must always return sensible, grounded
suggestions - regardless of whether the ML stack is installed, whether the
LoRA adapter has been trained yet, or whether a given generation attempt
produces garbage. Every suggestion is anchored to a real curated project;
the LLM only ever adds explanatory text, and that text is validated before
use.
"""
from __future__ import annotations

from typing import List

from backend import llm
from backend.catalog import Catalog, get_catalog
from backend.retrieval import ScoredProject, top_candidates
from backend.schemas import Suggestion, SuggestRequest, SuggestResponse


def _validate_llm_output(result: dict, allowed_ids: set, catalog: Catalog) -> bool:
    rationale = result.get("rationale", "")
    buy_next = result.get("buy_next", "")
    if not isinstance(rationale, str) or not isinstance(buy_next, str):
        return False
    if not (10 <= len(rationale) <= 600):
        return False
    if not (3 <= len(buy_next) <= 200):
        return False

    combined_text = f"{rationale} {buy_next}"

    # No-hallucination guard: flag if the model name-drops (by name OR alias,
    # via word-boundary matching so short aliases can't false-positive on
    # substrings of unrelated words) any catalog component that wasn't in
    # this project's have/missing lists.
    mentioned_ids = catalog.find_mentioned_component_ids(combined_text)
    if not mentioned_ids.issubset(allowed_ids):
        return False
    return True


def _template_rationale(sp: ScoredProject, have_names: List[str], missing_names: List[str], request: SuggestRequest):
    project = sp.project
    have_n = len(sp.have_required)
    total_n = len(project["required_components"])

    pieces = [
        f"This {project['skill_level']}-level {project['category'].lower()} project "
        f"covers {have_n}/{total_n} required parts you already listed"
    ]
    if request.time_budget_hours:
        pieces.append(f"fits within roughly your {request.time_budget_hours:g}h time budget")
    if request.interests.strip():
        pieces.append(f"relates to your interest in \"{request.interests.strip()}\"")
    rationale = ", ".join(pieces) + "."

    if missing_names:
        buy_next = f"Next part worth getting: {missing_names[0]}."
    else:
        buy_next = "You already have every required part for this build."

    return rationale, buy_next


def _build_suggestion(sp: ScoredProject, request: SuggestRequest, catalog: Catalog) -> Suggestion:
    project = sp.project
    have_names = [catalog.component_name(c) for c in (sp.have_required + sp.have_optional)]
    missing_names = [catalog.component_name(c) for c in sp.missing_required]

    source = "template_fallback"
    rationale_text = None
    buy_next_text = None

    if llm.is_available():
        try:
            user_context = {
                "skill_level": request.skill_level,
                "interests": request.interests,
                "time_budget_hours": request.time_budget_hours,
            }
            result = llm.generate_rationale(user_context, project, have_names, missing_names)
            allowed_ids = set(sp.have_required + sp.have_optional + sp.missing_required)
            if _validate_llm_output(result, allowed_ids, catalog):
                rationale_text = result["rationale"]
                buy_next_text = result["buy_next"]
                source = "fine_tuned_llm"
        except Exception:
            # any failure (model error, bad JSON, validation failure) -> fallback
            pass

    if rationale_text is None:
        rationale_text, buy_next_text = _template_rationale(sp, have_names, missing_names, request)

    return Suggestion(
        project_id=project["id"],
        title=project["title"],
        category=project["category"],
        skill_level=project["skill_level"],
        time_hours=project["time_hours"],
        match_score=sp.total_score,
        have_components=have_names,
        missing_components=missing_names,
        optional_components_owned=[catalog.component_name(c) for c in sp.have_optional],
        description=project["description"],
        learning_outcomes=project["learning_outcomes"],
        rationale=f"{rationale_text} {buy_next_text}".strip(),
        source=source,
    )


def get_suggestions(request: SuggestRequest, k: int = 3) -> SuggestResponse:
    catalog = get_catalog()
    available = catalog.normalize_many(request.available_hardware)

    scored = top_candidates(
        skill_level=request.skill_level,
        available_components=available,
        time_budget_hours=request.time_budget_hours,
        interests=request.interests,
        k=k,
        catalog=catalog,
    )

    suggestions = [_build_suggestion(sp, request, catalog) for sp in scored]

    engine_mode = "fine_tuned_llm" if any(s.source == "fine_tuned_llm" for s in suggestions) else "template_fallback"
    notes = None
    if engine_mode == "template_fallback":
        reason = llm.unavailable_reason() if not llm.is_available() else "model output failed validation"
        notes = f"Running in template fallback mode ({reason})."

    return SuggestResponse(suggestions=suggestions, engine_mode=engine_mode, notes=notes)
