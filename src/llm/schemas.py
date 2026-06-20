"""Pydantic schemas for the LLM pipeline structured output (Этап 5).

These are the response_models `instructor` validates each LLM call against, so the
pipeline never parses free-form text. See wiki/LLM_Pipeline.md.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Step 1 — normalized facts extracted from a resume (no invented data)."""

    title: str = Field(description="Краткое название профиля/роли кандидата")
    specializations: list[str] = Field(default_factory=list)
    years_experience: float | None = Field(
        default=None, description="Общий стаж в годах, если выводится из резюме")
    key_skills: list[str] = Field(
        default_factory=list, description="Навыки как отдельные технологии/компетенции")
    last_position: str | None = None
    education: str | None = None
    summary: str = Field(description="1–2 предложения нормализованного резюме профиля")


class Gap(BaseModel):
    """A single missing/weak requirement and how critical it is."""

    requirement: str
    severity: Literal["critical", "minor"] = Field(
        description="critical — без этого скорее не пригласят; minor — желательно")


class GapAnalysis(BaseModel):
    """Step 2 — match of profile vs vacancy with explicit reasoning (CoT)."""

    strengths: list[str] = Field(default_factory=list, description="Совпадения с требованиями")
    gaps: list[Gap] = Field(default_factory=list, description="Отсутствующие/слабые требования")
    reasoning: str = Field(description="Пошаговое рассуждение: совпадения → gaps → критичность")


class MatchVerdict(BaseModel):
    """Step 3 — final 0–100 score, recommendation and short explanation."""

    score: int = Field(ge=0, le=100, description="Соответствие на ПЕРВОЕ собеседование, 0–100")
    recommendation: Literal["invite", "consider", "reject"]
    explanation: str = Field(description="2–3 предложения: почему подходит/не подходит")


class CandidateEvaluation(BaseModel):
    """Full pipeline output for one (vacancy, resume) pair."""

    vacancy_id: str
    resume_id: str
    profile: CandidateProfile
    gap_analysis: GapAnalysis
    verdict: MatchVerdict
