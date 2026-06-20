"""Этап 4 — build the offline evaluation set from the labelled pairs.

Ground truth = pairs.parquet (vacancy_id, resume_id, target). For each vacancy
with >=1 target=1 resume, the relevant set is its target=1 resume_ids. These
resumes are real entities in the Qdrant index, so retrieval can surface them
(wiki/Евалюация.md).

Outputs eval/results/eval_set.parquet with columns:
    vacancy_id, relevant_ids (list[str]), n_relevant, n_labelled

Run:  python -m eval.build_eval_set                 # sample EVAL_SAMPLE_SIZE
      python -m eval.build_eval_set --limit 0       # all vacancies with positives
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config


def build_eval_set(limit: int = config.EVAL_SAMPLE_SIZE, seed: int = config.EVAL_SEED) -> pd.DataFrame:
    """Return a DataFrame of vacancies with their relevant resume_ids."""
    pairs = pd.read_parquet(config.PAIRS_PARQUET)
    pairs[config.VACANCY_ID] = pairs[config.VACANCY_ID].astype(str)
    pairs[config.RESUME_ID] = pairs[config.RESUME_ID].astype(str)

    n_labelled = pairs.groupby(config.VACANCY_ID).size().rename("n_labelled")
    positives = pairs[pairs[config.TARGET] == 1]
    relevant = (
        positives.groupby(config.VACANCY_ID)[config.RESUME_ID]
        .agg(lambda s: sorted(set(s)))
        .rename("relevant_ids")
    )

    eval_set = relevant.to_frame()
    eval_set["n_relevant"] = eval_set["relevant_ids"].str.len()
    eval_set = eval_set.join(n_labelled, on=config.VACANCY_ID).reset_index()

    total_with_pos = len(eval_set)
    if limit and limit > 0 and limit < total_with_pos:
        eval_set = eval_set.sample(n=limit, random_state=seed).reset_index(drop=True)

    print(f"vacancies with >=1 positive: {total_with_pos:,}")
    print(f"sampled for eval:            {len(eval_set):,} (seed={seed})")
    print(f"relevant per vacancy:        "
          f"mean={eval_set.n_relevant.mean():.1f} "
          f"median={eval_set.n_relevant.median():.0f} "
          f"min={eval_set.n_relevant.min()} max={eval_set.n_relevant.max()}")
    print(f"labelled candidates/vacancy: "
          f"mean={eval_set.n_labelled.mean():.1f} median={eval_set.n_labelled.median():.0f}")
    return eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline eval set from pairs.parquet.")
    parser.add_argument("--limit", type=int, default=config.EVAL_SAMPLE_SIZE,
                        help="how many vacancies to sample (0 = all with positives)")
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    args = parser.parse_args()

    eval_set = build_eval_set(limit=args.limit, seed=args.seed)
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_set.to_parquet(config.EVAL_SET_PARQUET, index=False)
    print(f"\nsaved -> {config.EVAL_SET_PARQUET}")


if __name__ == "__main__":
    main()
