"""Этап 3 — retrieval of resumes for a given vacancy.

Three modes (compared later in eval/, wiki/Евалюация.md):
  * dense  — Qdrant cosine search using the vacancy's embedding.
  * bm25   — lexical search over the resume corpus (rank_bm25).
  * hybrid — RRF fusion of dense + bm25 rankings.

Public entry point:
    search_resumes(vacancy_id, top_k=20, filters=None, mode="dense")
        -> list[(resume_id, score)]

Payload filters (city / experience / applicant status / age) are resolved once
against Qdrant and reused by every mode, so all three see the same candidate set.
See wiki/Hybrid_Search.md and wiki/Векторная_БД.md.

CLI smoke test:
    python -m src.vectorstore.search --vacancy_id 126167948 --mode hybrid
"""
from __future__ import annotations

import argparse
import pickle

import pandas as pd
from qdrant_client import models

from src import config
from src.data.preprocessing import build_resume_text, build_vacancy_text
from src.vectorstore import client as vsclient
from src.vectorstore import text_norm


def _tokenize(text: str) -> list[str]:
    """BM25 tokenizer — lemmatized or simple per config.BM25_LEMMATIZE."""
    if config.BM25_LEMMATIZE:
        return text_norm.normalize_tokens(text)
    return text_norm.simple_tokens(text)


class ResumeSearcher:
    """Holds the Qdrant client, resume/vacancy frames and a cached BM25 index."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.qc = vsclient.get_client()
        self._resumes: pd.DataFrame | None = None
        self._vacancies: pd.DataFrame | None = None
        self._bm25 = None
        self._bm25_ids: list[str] | None = None  # resume_id aligned with corpus order

    # --- lazy data ---------------------------------------------------------
    @property
    def resumes(self) -> pd.DataFrame:
        if self._resumes is None:
            self._resumes = pd.read_parquet(config.RESUMES_PARQUET)
            self._resumes[config.RESUME_ID] = self._resumes[config.RESUME_ID].astype(str)
        return self._resumes

    @property
    def vacancies(self) -> pd.DataFrame:
        if self._vacancies is None:
            self._vacancies = pd.read_parquet(config.VACANCIES_PARQUET)
            self._vacancies[config.VACANCY_ID] = self._vacancies[config.VACANCY_ID].astype(str)
        return self._vacancies

    # --- vacancy helpers ---------------------------------------------------
    def _vacancy_vector(self, vacancy_id):
        recs = self.qc.retrieve(
            config.COLLECTION_VACANCIES, ids=[int(vacancy_id)], with_vectors=True
        )
        if not recs:
            raise KeyError(f"vacancy_id {vacancy_id} not found in Qdrant")
        return recs[0].vector

    def _vacancy_text(self, vacancy_id) -> str:
        row = self.vacancies.loc[self.vacancies[config.VACANCY_ID] == str(vacancy_id)]
        if row.empty:
            raise KeyError(f"vacancy_id {vacancy_id} not found in parquet")
        return build_vacancy_text(row).iloc[0]

    # --- filters -----------------------------------------------------------
    @staticmethod
    def _build_qdrant_filter(filters: dict | None):
        """Translate a plain filter dict into a Qdrant Filter (or None).

        Supported keys: resume_area, resume_applicant_status (exact match);
        min_experience_months, max_age (range).
        """
        if not filters:
            return None
        must = []
        if "resume_area" in filters:
            must.append(models.FieldCondition(
                key="resume_area", match=models.MatchValue(value=filters["resume_area"])))
        if "resume_applicant_status" in filters:
            must.append(models.FieldCondition(
                key="resume_applicant_status",
                match=models.MatchValue(value=filters["resume_applicant_status"])))
        if "min_experience_months" in filters:
            must.append(models.FieldCondition(
                key="resume_experience_months",
                range=models.Range(gte=filters["min_experience_months"])))
        if "max_age" in filters:
            must.append(models.FieldCondition(
                key="resume_age", range=models.Range(lte=filters["max_age"])))
        return models.Filter(must=must) if must else None

    def _allowed_ids(self, qfilter) -> set[str] | None:
        """Return the set of resume_ids passing the filter, or None for 'no filter'."""
        if qfilter is None:
            return None
        allowed, offset = set(), None
        while True:
            points, offset = self.qc.scroll(
                config.COLLECTION_RESUMES, scroll_filter=qfilter,
                limit=4096, offset=offset, with_payload=["resume_id"], with_vectors=False)
            allowed.update(str(p.payload["resume_id"]) for p in points)
            if offset is None:
                break
        return allowed

    # --- BM25 --------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if config.BM25_RESUMES_PKL.exists():
            with open(config.BM25_RESUMES_PKL, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("version") == config.BM25_NORM_VERSION:
                self._bm25, self._bm25_ids = blob["bm25"], blob["ids"]
                return
            print("BM25 cache outdated (tokenization changed) — rebuilding ...", flush=True)
        from rank_bm25 import BM25Okapi

        norm = "lemmatized" if config.BM25_LEMMATIZE else "simple"
        print(f"Building BM25 index over resumes ({norm} tokenization, one-time) ...", flush=True)
        texts = build_resume_text(self.resumes)
        corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_ids = self.resumes[config.RESUME_ID].tolist()
        with open(config.BM25_RESUMES_PKL, "wb") as fh:
            pickle.dump({"version": config.BM25_NORM_VERSION,
                         "bm25": self._bm25, "ids": self._bm25_ids}, fh)

    def _embed_query(self, text: str):
        """Embed an arbitrary vacancy text on the fly (for ad-hoc vacancies)."""
        return vsclient.embed_texts([text], show_progress=False)[0].tolist()

    # --- single-mode rankers (return ranked list of resume_id) -------------
    def _dense_by_vector(self, vector, limit, qfilter) -> list[tuple[str, float]]:
        hits = self.qc.query_points(
            config.COLLECTION_RESUMES, query=vector,
            limit=limit, query_filter=qfilter, with_payload=["resume_id"]).points
        return [(str(h.payload["resume_id"]), float(h.score)) for h in hits]

    def _dense_ranked(self, vacancy_id, limit, qfilter) -> list[tuple[str, float]]:
        return self._dense_by_vector(self._vacancy_vector(vacancy_id), limit, qfilter)

    def _bm25_by_text(self, text, limit, allowed: set[str] | None) -> list[tuple[str, float]]:
        self._ensure_bm25()
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        if allowed is not None:
            ranked = [(rid, s) for rid, s in ranked if rid in allowed]
        return [(rid, float(s)) for rid, s in ranked[:limit]]

    def _bm25_ranked(self, vacancy_id, limit, allowed: set[str] | None) -> list[tuple[str, float]]:
        return self._bm25_by_text(self._vacancy_text(vacancy_id), limit, allowed)

    @staticmethod
    def _rrf(rankings: list[list[tuple[str, float]]], top_k: int) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of several ranked lists -> top_k (id, rrf_score)."""
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, (rid, _score) in enumerate(ranking, start=1):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (config.RRF_K + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # --- public API --------------------------------------------------------
    def search_resumes(
        self,
        vacancy_id,
        top_k: int = config.DEFAULT_TOP_K,
        filters: dict | None = None,
        mode: str = "dense",
    ) -> list[tuple[str, float]]:
        """Return top_k (resume_id, score) for a vacancy in the requested mode."""
        qfilter = self._build_qdrant_filter(filters)

        if mode == "dense":
            return self._dense_ranked(vacancy_id, top_k, qfilter)
        if mode == "bm25":
            return self._bm25_ranked(vacancy_id, top_k, self._allowed_ids(qfilter))
        if mode == "hybrid":
            pool = max(top_k, config.HYBRID_CANDIDATE_POOL)
            dense = self._dense_ranked(vacancy_id, pool, qfilter)
            bm25 = self._bm25_ranked(vacancy_id, pool, self._allowed_ids(qfilter))
            return self._rrf([dense, bm25], top_k)
        raise ValueError(f"unknown mode {mode!r} (expected dense|bm25|hybrid)")

    def search_by_text(
        self,
        vacancy_text: str,
        top_k: int = config.DEFAULT_TOP_K,
        filters: dict | None = None,
        mode: str = "dense",
    ) -> list[tuple[str, float]]:
        """Like search_resumes but for an ad-hoc vacancy given as raw text.

        Embeds the text on the fly (loads the embedding model) instead of
        retrieving a stored vacancy vector — used for user-entered vacancies in
        the UI (wiki/LLM_Pipeline.md, Этап 6)."""
        qfilter = self._build_qdrant_filter(filters)

        if mode == "bm25":
            return self._bm25_by_text(vacancy_text, top_k, self._allowed_ids(qfilter))
        vector = self._embed_query(vacancy_text)
        if mode == "dense":
            return self._dense_by_vector(vector, top_k, qfilter)
        if mode == "hybrid":
            pool = max(top_k, config.HYBRID_CANDIDATE_POOL)
            dense = self._dense_by_vector(vector, pool, qfilter)
            bm25 = self._bm25_by_text(vacancy_text, pool, self._allowed_ids(qfilter))
            return self._rrf([dense, bm25], top_k)
        raise ValueError(f"unknown mode {mode!r} (expected dense|bm25|hybrid)")


# Module-level convenience (lazy singleton) ---------------------------------
_DEFAULT_SEARCHER: ResumeSearcher | None = None


def _searcher() -> ResumeSearcher:
    global _DEFAULT_SEARCHER
    if _DEFAULT_SEARCHER is None:
        _DEFAULT_SEARCHER = ResumeSearcher()
    return _DEFAULT_SEARCHER


def search_resumes(vacancy_id, top_k=config.DEFAULT_TOP_K, filters=None, mode="dense"):
    """Functional wrapper around a shared ResumeSearcher (see spec Этап 3)."""
    return _searcher().search_resumes(vacancy_id, top_k=top_k, filters=filters, mode=mode)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve resumes for a vacancy.")
    parser.add_argument("--vacancy_id", required=True)
    parser.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--area", help="filter: resume_area (exact city match)")
    args = parser.parse_args()

    filters = {"resume_area": args.area} if args.area else None
    searcher = ResumeSearcher()
    rtitle = dict(zip(searcher.resumes[config.RESUME_ID], searcher.resumes["resume_title"].astype(str)))

    print(f"vacancy {args.vacancy_id} | mode={args.mode} | top_k={args.top_k}"
          + (f" | area={args.area}" if args.area else ""))
    for rid, score in searcher.search_resumes(args.vacancy_id, args.top_k, filters, args.mode):
        print(f"  {score:.4f}  {rid:>10}  {rtitle.get(rid, '?')[:60]}")


if __name__ == "__main__":
    _main()
