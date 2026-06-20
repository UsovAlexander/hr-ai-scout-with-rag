"""Этап 4 — retrieval metrics, implemented from scratch.

Deliberately NOT using a ready-made RAG-eval library (project_spec §7): the point
is to show the formulas explicitly. See wiki/Евалюация.md.

Every function takes:
    retrieved : list[str]   ordered list of retrieved resume_ids (rank 1 first)
    relevant  : set[str]    resume_ids with target=1 for the vacancy (ground truth)
    k         : int         cutoff

Relevance is binary (target=1 -> relevant). MRR uses the full retrieved list;
the other metrics use the top-k slice.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _hits(retrieved: Sequence[str], relevant: set[str], k: int) -> list[int]:
    """Binary relevance of the top-k retrieved items (1 = relevant)."""
    return [1 if rid in relevant else 0 for rid in retrieved[:k]]


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k that is relevant:  TP@k / k."""
    if k <= 0:
        return 0.0
    return sum(_hits(retrieved, relevant, k)) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in the top-k:  TP@k / |relevant|."""
    if not relevant:
        return 0.0
    return sum(_hits(retrieved, relevant, k)) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item; 0 if none retrieved (this is MRR per query)."""
    for rank, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with log2(rank+1) discount, ranks starting at 1."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized DCG@k for binary relevance: DCG@k / ideal DCG@k."""
    dcg = dcg_at_k(_hits(retrieved, relevant, k))
    n_ideal = min(len(relevant), k)
    idcg = dcg_at_k([1] * n_ideal)
    return dcg / idcg if idcg > 0 else 0.0


def compute_all(retrieved: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    """All metrics for a single query."""
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


def macro_average(per_query: Iterable[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries (macro-average over vacancies)."""
    rows = list(per_query)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(r[key] for r in rows) / len(rows) for key in keys}


def _self_test() -> None:
    """Tiny sanity check of the formulas against hand-computed values."""
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d"}
    k = 5
    assert precision_at_k(retrieved, relevant, k) == 2 / 5
    assert recall_at_k(retrieved, relevant, k) == 2 / 2
    assert reciprocal_rank(retrieved, relevant) == 1 / 2  # first relevant at rank 2
    # DCG = 1/log2(3) + 1/log2(5); IDCG (2 relevant) = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(ndcg_at_k(retrieved, relevant, k) - dcg / idcg) < 1e-12
    # no relevant retrieved
    assert reciprocal_rank(["x", "y"], relevant) == 0.0
    assert recall_at_k(["x"], set(), 5) == 0.0
    print("retrieval_metrics self-test: OK")


if __name__ == "__main__":
    _self_test()
