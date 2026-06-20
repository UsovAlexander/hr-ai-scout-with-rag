"""Этап 2 — build Qdrant collections `resumes` and `vacancies`.

Reads the deduplicated parquet artifacts (Этап 1), embeds the concatenated text
per entity (project_spec §4.2), and upserts points with the payload fields used
for filtering during retrieval (wiki/Векторная_БД.md).

Entities are indexed, not table rows. Point id = the integer entity id, so the
real resume_id / vacancy_id is also kept in payload for convenience.

Run:  python -m src.vectorstore.indexer            # both collections
      python -m src.vectorstore.indexer --only resumes
"""
from __future__ import annotations

import argparse
import math

import pandas as pd
from qdrant_client import models

from src import config
from src.data.preprocessing import build_resume_text, build_vacancy_text
from src.vectorstore import client as vsclient

UPSERT_BATCH = 256

# Payload fields per project_spec §4.2. `source` -> `payload_key` mapping;
# numeric fields are coerced so range filters work in Qdrant.
RESUME_PAYLOAD = {
    "resume_id": "resume_id",
    "resume_location": "resume_area",  # spec: resume_area = resume_location
    "resume_experience_months": "resume_experience_months",
    "resume_salary": "resume_salary",
    "resume_applicant_status": "resume_applicant_status",
    "resume_age": "resume_age",
}
RESUME_NUMERIC = {"resume_experience_months", "resume_salary", "resume_age"}

VACANCY_PAYLOAD = {
    "vacancy_id": "vacancy_id",
    "vacancy_area": "vacancy_area",
    "vacancy_employment": "vacancy_employment",
    "vacancy_schedule": "vacancy_schedule",
    "vacancy_salary_from": "vacancy_salary_from",
    "vacancy_salary_to": "vacancy_salary_to",
    "vacancy_salary_currency": "vacancy_salary_currency",
}
VACANCY_NUMERIC = {"vacancy_salary_from", "vacancy_salary_to"}


def _clean(value, numeric: bool):
    """Normalize a payload value: drop NaN, coerce numerics, keep clean strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if numeric:
        num = pd.to_numeric(value, errors="coerce")
        return None if pd.isna(num) else (int(num) if float(num).is_integer() else float(num))
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _build_payload(row: pd.Series, mapping: dict, numeric: set) -> dict:
    payload = {}
    for src_col, key in mapping.items():
        cleaned = _clean(row.get(src_col), numeric=src_col in numeric)
        if cleaned is not None:
            payload[key] = cleaned
    return payload


def _recreate_collection(qc, name: str, dim: int) -> None:
    """(Re)create a collection with Cosine distance and the model's vector size."""
    if qc.collection_exists(name):
        qc.delete_collection(name)
    qc.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )


def index_collection(
    qc,
    name: str,
    df: pd.DataFrame,
    id_col: str,
    texts: pd.Series,
    payload_map: dict,
    numeric: set,
) -> int:
    """Embed `texts` and upsert all rows of `df` into collection `name`."""
    dim = vsclient.embedding_dim()
    _recreate_collection(qc, name, dim)

    print(f"[{name}] embedding {len(df):,} entities ...", flush=True)
    vectors = vsclient.embed_texts(texts.tolist())

    points = []
    total = 0
    for (_, row), vector in zip(df.iterrows(), vectors):
        point_id = int(pd.to_numeric(row[id_col]))
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=_build_payload(row, payload_map, numeric),
            )
        )
        if len(points) >= UPSERT_BATCH:
            qc.upsert(collection_name=name, points=points)
            total += len(points)
            points = []
    if points:
        qc.upsert(collection_name=name, points=points)
        total += len(points)

    print(f"[{name}] upserted {total:,} points (dim={dim}).", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Index resumes/vacancies into Qdrant.")
    parser.add_argument("--only", choices=["resumes", "vacancies"], help="Index just one collection.")
    args = parser.parse_args()

    qc = vsclient.get_client()

    if args.only != "vacancies":
        resumes = pd.read_parquet(config.RESUMES_PARQUET)
        index_collection(
            qc,
            config.COLLECTION_RESUMES,
            resumes,
            config.RESUME_ID,
            build_resume_text(resumes),
            RESUME_PAYLOAD,
            RESUME_NUMERIC,
        )

    if args.only != "resumes":
        vacancies = pd.read_parquet(config.VACANCIES_PARQUET)
        index_collection(
            qc,
            config.COLLECTION_VACANCIES,
            vacancies,
            config.VACANCY_ID,
            build_vacancy_text(vacancies),
            VACANCY_PAYLOAD,
            VACANCY_NUMERIC,
        )

    print("\n=== Indexing done ===")
    for coll in (config.COLLECTION_RESUMES, config.COLLECTION_VACANCIES):
        if qc.collection_exists(coll):
            print(f"{coll}: {qc.count(coll).count:,} points")


if __name__ == "__main__":
    main()
