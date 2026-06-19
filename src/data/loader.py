"""Этап 1 — load total_df.csv, deduplicate entities, persist parquet artifacts.

The CSV is a cartesian product of vacancy × resume (~2GB, 332k labelled rows),
so the same vacancy/resume appears in many rows. We index *entities*, not table
rows (wiki/Датасет.md), therefore:

  * vacancies  -> dedup by vacancy_id  -> vacancies.parquet
  * resumes    -> dedup by resume_id   -> resumes.parquet
  * pairs      -> (vacancy_id, resume_id, target) kept as the eval ground truth
                  -> pairs.parquet  (NEVER discard this)

Reads the file in chunks to stay memory-bounded.

Run:  python -m src.data.loader
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config


def _read_chunks(csv_path, usecols, chunksize):
    """Yield deduped-per-chunk frames for the given columns."""
    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        dtype=str,  # read as text; numeric coercion happens downstream per need
    )
    for chunk in reader:
        yield chunk


def load_and_dedup(
    csv_path=config.RAW_CSV,
    chunksize: int = config.READ_CHUNKSIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stream the CSV once and return (vacancies, resumes, pairs) frames."""
    vacancy_parts: list[pd.DataFrame] = []
    resume_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []

    seen_vacancies: set[str] = set()
    seen_resumes: set[str] = set()
    n_rows = 0

    needed = sorted(set(config.VACANCY_COLUMNS) | set(config.RESUME_COLUMNS) | {config.TARGET})

    for i, chunk in enumerate(_read_chunks(csv_path, needed, chunksize)):
        n_rows += len(chunk)

        # Vacancies: keep first occurrence of each unseen vacancy_id.
        vac = chunk[config.VACANCY_COLUMNS].drop_duplicates(subset=config.VACANCY_ID)
        vac = vac[~vac[config.VACANCY_ID].isin(seen_vacancies)]
        if not vac.empty:
            seen_vacancies.update(vac[config.VACANCY_ID].tolist())
            vacancy_parts.append(vac)

        # Resumes: same idea.
        res = chunk[config.RESUME_COLUMNS].drop_duplicates(subset=config.RESUME_ID)
        res = res[~res[config.RESUME_ID].isin(seen_resumes)]
        if not res.empty:
            seen_resumes.update(res[config.RESUME_ID].tolist())
            resume_parts.append(res)

        # Pairs: every labelled row is ground truth.
        pair_parts.append(chunk[config.PAIR_COLUMNS])

        print(
            f"  chunk {i:>3}: rows={n_rows:>9,} "
            f"unique_vacancies={len(seen_vacancies):>6,} "
            f"unique_resumes={len(seen_resumes):>7,}",
            flush=True,
        )

    vacancies = pd.concat(vacancy_parts, ignore_index=True)
    resumes = pd.concat(resume_parts, ignore_index=True)
    pairs = pd.concat(pair_parts, ignore_index=True)
    pairs[config.TARGET] = pd.to_numeric(pairs[config.TARGET], errors="coerce").astype("Int64")

    return vacancies, resumes, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Load + dedup total_df.csv into parquet artifacts.")
    parser.add_argument("--csv", default=str(config.RAW_CSV), help="Path to total_df.csv")
    parser.add_argument("--chunksize", type=int, default=config.READ_CHUNKSIZE)
    args = parser.parse_args()

    print(f"Reading {args.csv} (chunksize={args.chunksize:,}) ...", flush=True)
    vacancies, resumes, pairs = load_and_dedup(args.csv, args.chunksize)

    vacancies.to_parquet(config.VACANCIES_PARQUET, index=False)
    resumes.to_parquet(config.RESUMES_PARQUET, index=False)
    pairs.to_parquet(config.PAIRS_PARQUET, index=False)

    target_counts = pairs[config.TARGET].value_counts(dropna=False).to_dict()
    print("\n=== Done ===")
    print(f"vacancies: {len(vacancies):>7,} -> {config.VACANCIES_PARQUET}")
    print(f"resumes:   {len(resumes):>7,} -> {config.RESUMES_PARQUET}")
    print(f"pairs:     {len(pairs):>7,} -> {config.PAIRS_PARQUET}")
    print(f"target distribution: {target_counts}")


if __name__ == "__main__":
    main()
