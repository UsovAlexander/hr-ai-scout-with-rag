"""Qdrant client + embedding-model helpers.

Thin wrappers so the rest of the codebase doesn't repeat connection/model
boilerplate. See wiki/Векторная_БД.md.
"""
from __future__ import annotations

import functools

from qdrant_client import QdrantClient

from src import config


def get_client() -> QdrantClient:
    """Return a Qdrant client pointed at the local Docker instance."""
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def _pick_device() -> str:
    """Prefer Apple MPS, then CUDA, else CPU (project_spec Этап 2)."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@functools.lru_cache(maxsize=2)
def get_embedder(model_name: str = config.EMBEDDING_MODEL):
    """Load (and cache) a sentence-transformers model on the best device.

    Cached so repeated calls in indexer/search reuse one loaded model.
    """
    from sentence_transformers import SentenceTransformer

    device = _pick_device()
    print(f"Loading embedder '{model_name}' on device={device} ...", flush=True)
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts, model_name: str = config.EMBEDDING_MODEL):
    """Embed a list of strings, returning a list[list[float]].

    Batched per project_spec (batch_size=64). Normalized so cosine == dot,
    which matches the Cosine distance we configure the collections with.
    """
    model = get_embedder(model_name)
    vectors = model.encode(
        list(texts),
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors


def embedding_dim(model_name: str = config.EMBEDDING_MODEL) -> int:
    """Vector size of the configured embedding model."""
    return get_embedder(model_name).get_sentence_embedding_dimension()
