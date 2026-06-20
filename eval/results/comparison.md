# Retrieval comparison — dense / BM25 / hybrid

_Generated 2026-06-20 13:59 by `python -m eval.run_eval`._

- Embedding model: `cointegrated/LaBSE-en-ru` (dim 768, Cosine)
- Vacancies evaluated (macro-average): **500**
- Metrics @k = **10**, retrieval depth = 100, seed = 42
- Ground truth: `target = 1` resume_ids per vacancy (binary relevance)

| mode | recall@10 | precision@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| random | 0.0006 | 0.0004 | 0.0014 | 0.0004 |
| bm25 | 0.4349 | 0.3128 | 0.7480 | 0.4643 |
| dense | 0.1120 | 0.0772 | 0.2737 | 0.1151 |
| hybrid | 0.2427 | 0.1732 | 0.5494 | 0.2677 |

## How to read this

- **recall@10** — share of the vacancy's `target=1` resumes landing in the top-10.
  The headline number: did retrieval surface the people HR actually shortlisted?
- **mrr** — how high the first relevant resume sits (computed over depth 100).
- **ndcg@10** — ranking quality with positional discount.
- **precision@10** is reported but is a **lower bound**: retrieval runs over all
  20,845 resumes while only ~97 resumes per vacancy were
  ever labelled, so many top-10 resumes are simply *unlabelled* for this vacancy
  (treated here as non-relevant), not truly negative. See wiki/Евалюация.md.

## Findings

- **Best mode by recall@10: `bm25`** (0.435 recall, 0.464 ndcg).
- Naive RRF `hybrid` does NOT beat the best single mode (bm25) on recall@10 (0.243 vs 0.435) — equal-weight fusion dilutes the stronger lexical signal. Motivates **weighted** fusion as a follow-up.
- Lexical `bm25` >> `dense` here: recruiter shortlisting on this corpus correlates with explicit skill/term overlap more than with embedding similarity over the long concatenated resume text.
- `random` ≈ 0.0006 matches the theoretical floor (~k/|corpus|), confirming the metric plumbing is correct.
