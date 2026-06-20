"""Этап 5 — LLM pipeline: extraction -> gap analysis -> scoring.

Runs on top of the top-K candidates from retrieval (src/vectorstore/search.py).
Uses an OpenAI-compatible endpoint (Groq by default) + `instructor` for
structured output validated against src/llm/schemas.py. Prompts are versioned
.txt templates in src/llm/prompts/ (not hardcoded). See wiki/LLM_Pipeline.md.

CLI:
    export GROQ_API_KEY=...
    python -m src.llm.pipeline --vacancy_id 126167948 --resume_id 6969174
"""
from __future__ import annotations

import argparse
import functools
import os

import pandas as pd

from src import config
from src.data.preprocessing import build_resume_text, build_vacancy_text
from src.llm.schemas import (
    CandidateEvaluation,
    CandidateProfile,
    GapAnalysis,
    MatchVerdict,
)


# --- prompt loading --------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


def _render(template_name: str, **fields: str) -> str:
    """Fill {{token}} placeholders. Uses replace (not str.format) so braces in
    resume/vacancy text don't break rendering."""
    text = _load_prompt(template_name)
    for key, value in fields.items():
        text = text.replace("{{" + key + "}}", value)
    return text


# --- LLM client ------------------------------------------------------------
def _resolve_api_key() -> str | None:
    """Read the API key from the environment or the project .env file.

    Accepts both GROQ_API_KEY and the lowercase groq_api_key.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(config.PROJECT_ROOT / ".env")
    except ModuleNotFoundError:
        pass
    env = config.LLM_API_KEY_ENV
    return os.environ.get(env) or os.environ.get(env.lower())


@functools.lru_cache(maxsize=1)
def get_llm():
    """Instructor-wrapped OpenAI client pointed at the configured endpoint."""
    import instructor
    from openai import OpenAI

    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            f"Set {config.LLM_API_KEY_ENV} (or groq_api_key) in the environment "
            f"or in {config.PROJECT_ROOT / '.env'} to use the LLM layer."
        )
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=api_key)
    # JSON mode is the safe structured-output mode across Groq models.
    return instructor.from_openai(client, mode=instructor.Mode.JSON)


def _complete(prompt: str, response_model, model: str):
    return get_llm().chat.completions.create(
        model=model,
        temperature=config.LLM_TEMPERATURE,
        response_model=response_model,
        messages=[{"role": "user", "content": prompt}],
    )


# --- pipeline steps --------------------------------------------------------
def extract_profile(resume_text: str, model: str = config.LLM_MODEL) -> CandidateProfile:
    """Step 1 — normalize a resume into a structured profile."""
    return _complete(_render("extraction.txt", resume_text=resume_text),
                     CandidateProfile, model)


def analyze_gaps(profile: CandidateProfile, vacancy_text: str,
                 model: str = config.LLM_MODEL) -> GapAnalysis:
    """Step 2 — match profile vs vacancy with explicit reasoning."""
    prompt = _render("gap_analysis.txt", vacancy_text=vacancy_text,
                     profile=profile.model_dump_json(indent=2))
    return _complete(prompt, GapAnalysis, model)


def score_candidate(profile: CandidateProfile, gaps: GapAnalysis, vacancy_text: str,
                    model: str = config.LLM_MODEL) -> MatchVerdict:
    """Step 3 — final 0–100 score + recommendation + explanation."""
    prompt = _render("scoring.txt", vacancy_text=vacancy_text,
                     profile=profile.model_dump_json(indent=2),
                     gaps=gaps.model_dump_json(indent=2))
    return _complete(prompt, MatchVerdict, model)


# --- data access -----------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    resumes = pd.read_parquet(config.RESUMES_PARQUET)
    resumes[config.RESUME_ID] = resumes[config.RESUME_ID].astype(str)
    vacancies = pd.read_parquet(config.VACANCIES_PARQUET)
    vacancies[config.VACANCY_ID] = vacancies[config.VACANCY_ID].astype(str)
    return resumes, vacancies


def _resume_text(resume_id: str) -> str:
    resumes, _ = _frames()
    row = resumes.loc[resumes[config.RESUME_ID] == str(resume_id)]
    if row.empty:
        raise KeyError(f"resume_id {resume_id} not found")
    return build_resume_text(row).iloc[0]


def _vacancy_text(vacancy_id: str) -> str:
    _, vacancies = _frames()
    row = vacancies.loc[vacancies[config.VACANCY_ID] == str(vacancy_id)]
    if row.empty:
        raise KeyError(f"vacancy_id {vacancy_id} not found")
    return build_vacancy_text(row).iloc[0]


def evaluate_with_texts(vacancy_text: str, resume_text: str, vacancy_id, resume_id,
                        model: str = config.LLM_MODEL) -> CandidateEvaluation:
    """Full chain given the raw vacancy and resume texts."""
    profile = extract_profile(resume_text, model)
    gaps = analyze_gaps(profile, vacancy_text, model)
    verdict = score_candidate(profile, gaps, vacancy_text, model)
    return CandidateEvaluation(
        vacancy_id=str(vacancy_id), resume_id=str(resume_id),
        profile=profile, gap_analysis=gaps, verdict=verdict,
    )


def evaluate_candidate(vacancy_id, resume_id, model: str = config.LLM_MODEL) -> CandidateEvaluation:
    """Full chain for a stored vacancy x resume pair."""
    return evaluate_with_texts(_vacancy_text(vacancy_id), _resume_text(resume_id),
                               vacancy_id, resume_id, model)


def evaluate_custom_vacancy(vacancy_text: str, resume_id, model: str = config.LLM_MODEL
                            ) -> CandidateEvaluation:
    """Full chain for a user-entered vacancy text against a stored resume."""
    return evaluate_with_texts(vacancy_text, _resume_text(resume_id), "custom", resume_id, model)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM pipeline for one pair.")
    parser.add_argument("--vacancy_id", required=True)
    parser.add_argument("--resume_id", required=True)
    parser.add_argument("--model", default=config.LLM_MODEL)
    args = parser.parse_args()

    result = evaluate_candidate(args.vacancy_id, args.resume_id, args.model)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    _main()
