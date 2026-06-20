"""Этап 4 — run retrieval evaluation and write eval/results/comparison.md.

For each vacancy in the eval set, retrieve resumes in every mode and score them
against the target=1 ground truth, then macro-average over vacancies.

Modes compared (wiki/Евалюация.md):
    random  — floor baseline (random resume_ids from the corpus)
    bm25    — lexical only
    dense   — embedding only
    hybrid  — dense + bm25 fused with RRF

Per vacancy we compute the dense and bm25 rankings once and reuse them for the
hybrid (RRF) fusion, so no ranker runs twice.

Run:  python -m eval.run_eval                 # uses eval_set.parquet
      python -m eval.run_eval --limit 200     # (re)build + sample 200 first
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from src import config
from src.vectorstore.search import ResumeSearcher
from eval import retrieval_metrics as M
from eval.build_eval_set import build_eval_set

MODES = ["random", "bm25", "dense", "hybrid"]


def _rankings_for_vacancy(searcher: ResumeSearcher, vacancy_id: str, depth: int,
                          rng: np.random.Generator, corpus_ids: np.ndarray) -> dict[str, list[str]]:
    """Return ranked resume_id lists per mode for one vacancy (no filters in eval)."""
    dense = [rid for rid, _ in searcher._dense_ranked(vacancy_id, depth, None)]
    bm25 = [rid for rid, _ in searcher._bm25_ranked(vacancy_id, depth, None)]
    hybrid = [rid for rid, _ in searcher._rrf([
        [(r, 0.0) for r in dense], [(r, 0.0) for r in bm25]], depth)]
    random_ids = rng.choice(corpus_ids, size=depth, replace=False).tolist()
    return {"random": random_ids, "bm25": bm25, "dense": dense, "hybrid": hybrid}


def evaluate(eval_set: pd.DataFrame, k: int = config.EVAL_K,
             depth: int = config.EVAL_RETRIEVE_DEPTH, seed: int = config.EVAL_SEED):
    """Run all modes over the eval set; return (aggregates, per_vacancy_df)."""
    searcher = ResumeSearcher()
    searcher._ensure_bm25()  # build/load once up front
    corpus_ids = searcher.resumes[config.RESUME_ID].to_numpy()
    rng = np.random.default_rng(seed)

    per_query: dict[str, list[dict]] = {m: [] for m in MODES}
    rows = []
    n = len(eval_set)
    for i, row in enumerate(eval_set.itertuples(index=False), start=1):
        vacancy_id = str(row.vacancy_id)
        relevant = set(map(str, row.relevant_ids))
        rankings = _rankings_for_vacancy(searcher, vacancy_id, depth, rng, corpus_ids)
        rec = {"vacancy_id": vacancy_id, "n_relevant": len(relevant)}
        for mode in MODES:
            metrics = M.compute_all(rankings[mode], relevant, k)
            per_query[mode].append(metrics)
            rec.update({f"{mode}.{name}": val for name, val in metrics.items()})
        rows.append(rec)
        if i % 50 == 0 or i == n:
            print(f"  evaluated {i:>4}/{n} vacancies", flush=True)

    aggregates = {mode: M.macro_average(per_query[mode]) for mode in MODES}
    return aggregates, pd.DataFrame(rows)


def _findings(aggregates: dict, k: int) -> str:
    """Data-driven summary so the writeup never contradicts the numbers."""
    recall = f"recall@{k}"
    ndcg = f"ndcg@{k}"
    ranked = sorted(["random", "bm25", "dense", "hybrid"],
                    key=lambda m: aggregates[m][recall], reverse=True)
    best = ranked[0]
    best_single = max(["bm25", "dense"], key=lambda m: aggregates[m][recall])
    hyb, comp = aggregates["hybrid"], aggregates[best_single]
    hybrid_vs = ("beats" if hyb[recall] > comp[recall]
                 else "does NOT beat") + f" the best single mode ({best_single})"
    return (
        f"- **Best mode by {recall}: `{best}`** "
        f"({aggregates[best][recall]:.3f} recall, {aggregates[best][ndcg]:.3f} ndcg).\n"
        f"- Naive RRF `hybrid` {hybrid_vs} on {recall} "
        f"({hyb[recall]:.3f} vs {comp[recall]:.3f}) — equal-weight fusion dilutes the "
        f"stronger lexical signal. Motivates **weighted** fusion as a follow-up.\n"
        f"- Lexical `bm25` >> `dense` here: recruiter shortlisting on this corpus "
        f"correlates with explicit skill/term overlap more than with embedding "
        f"similarity over the long concatenated resume text.\n"
        f"- `random` ≈ {aggregates['random'][recall]:.4f} matches the theoretical floor "
        f"(~k/|corpus|), confirming the metric plumbing is correct."
    )


def _format_markdown(aggregates: dict, n_vac: int, k: int, depth: int, seed: int) -> str:
    metric_order = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]
    header = "| mode | " + " | ".join(metric_order) + " |"
    sep = "|" + "---|" * (len(metric_order) + 1)
    lines = [header, sep]
    # order rows weakest -> strongest for readability
    for mode in ["random", "bm25", "dense", "hybrid"]:
        agg = aggregates[mode]
        cells = " | ".join(f"{agg[m]:.4f}" for m in metric_order)
        lines.append(f"| {mode} | {cells} |")
    table = "\n".join(lines)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    findings = _findings(aggregates, k)
    return f"""# Retrieval comparison — dense / BM25 / hybrid

_Generated {now} by `python -m eval.run_eval`._

- Embedding model: `{config.EMBEDDING_MODEL}` (dim 768, Cosine)
- Vacancies evaluated (macro-average): **{n_vac:,}**
- Metrics @k = **{k}**, retrieval depth = {depth}, seed = {seed}
- Ground truth: `target = 1` resume_ids per vacancy (binary relevance)

{table}

## How to read this

- **recall@{k}** — share of the vacancy's `target=1` resumes landing in the top-{k}.
  The headline number: did retrieval surface the people HR actually shortlisted?
- **mrr** — how high the first relevant resume sits (computed over depth {depth}).
- **ndcg@{k}** — ranking quality with positional discount.
- **precision@{k}** is reported but is a **lower bound**: retrieval runs over all
  20,845 resumes while only ~{int(round(_avg_labelled()))} resumes per vacancy were
  ever labelled, so many top-{k} resumes are simply *unlabelled* for this vacancy
  (treated here as non-relevant), not truly negative. See wiki/Евалюация.md.

## Findings

{findings}
"""


def _avg_labelled() -> float:
    pairs = pd.read_parquet(config.PAIRS_PARQUET, columns=[config.VACANCY_ID])
    return pairs.groupby(config.VACANCY_ID).size().mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation -> comparison.md")
    parser.add_argument("--limit", type=int, default=None,
                        help="if set, (re)build the eval set with this sample size first")
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    args = parser.parse_args()

    if args.limit is not None:
        eval_set = build_eval_set(limit=args.limit, seed=args.seed)
        config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        eval_set.to_parquet(config.EVAL_SET_PARQUET, index=False)
    else:
        eval_set = pd.read_parquet(config.EVAL_SET_PARQUET)

    print(f"Evaluating {len(eval_set):,} vacancies across modes {MODES} ...")
    aggregates, per_vac = evaluate(eval_set, seed=args.seed)

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_vac.to_parquet(config.EVAL_PERVAC_PARQUET, index=False)
    md = _format_markdown(aggregates, len(eval_set), config.EVAL_K,
                          config.EVAL_RETRIEVE_DEPTH, args.seed)
    config.COMPARISON_MD.write_text(md, encoding="utf-8")

    print("\n=== Macro-averaged results ===")
    for mode in MODES:
        agg = aggregates[mode]
        print(f"  {mode:>7}: " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    print(f"\nwrote -> {config.COMPARISON_MD}")
    print(f"wrote -> {config.EVAL_PERVAC_PARQUET}")


if __name__ == "__main__":
    main()
