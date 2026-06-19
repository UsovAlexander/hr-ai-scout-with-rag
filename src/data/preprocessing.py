"""Assemble the text that gets embedded for each entity.

Concatenation rules come straight from project_spec §4.2 (wiki/Векторная_БД.md):
resumes and vacancies each glue a fixed set of fields into a single string.
"""
from __future__ import annotations

import pandas as pd

from src import config


def _join_fields(row: pd.Series, fields: list[str]) -> str:
    """Join the given fields of a row into one space-separated string.

    NaN / empty values are skipped so missing fields don't inject the literal
    "nan" into the embedded text.
    """
    parts = []
    for field in fields:
        value = row.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            parts.append(text)
    return " ".join(parts)


def build_resume_text(resumes: pd.DataFrame) -> pd.Series:
    """Return the embedding text for each resume row."""
    return resumes.apply(lambda r: _join_fields(r, config.RESUME_TEXT_FIELDS), axis=1)


def build_vacancy_text(vacancies: pd.DataFrame) -> pd.Series:
    """Return the embedding text for each vacancy row."""
    return vacancies.apply(lambda r: _join_fields(r, config.VACANCY_TEXT_FIELDS), axis=1)
