"""Central configuration for HR AI Scout.

Paths, column groups, embedding-model selection and Qdrant settings live here so
they can be swapped from one place (e.g. embedding model for ablation in eval/).
See wiki/Векторная_БД.md and wiki/Датасет.md.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_CSV = DATASET_DIR / "total_df.csv"

# Derived, deduplicated artifacts (Этап 1)
VACANCIES_PARQUET = DATASET_DIR / "vacancies.parquet"
RESUMES_PARQUET = DATASET_DIR / "resumes.parquet"
PAIRS_PARQUET = DATASET_DIR / "pairs.parquet"  # ground truth for eval/

# Cached BM25 index over the resume corpus (Этап 3)
BM25_RESUMES_PKL = DATASET_DIR / "bm25_resumes.pkl"

# --- CSV reading -----------------------------------------------------------
# total_df.csv is ~2GB → read in chunks, never load naively into RAM.
READ_CHUNKSIZE = 100_000

# --- Entity keys -----------------------------------------------------------
VACANCY_ID = "vacancy_id"
RESUME_ID = "resume_id"
TARGET = "target"

# --- Column groups ---------------------------------------------------------
# All vacancy / resume columns (used to slice + dedup, see data/loader.py).
VACANCY_COLUMNS = [
    "vacancy_id",
    "vacancy_name",
    "vacancy_area",
    "vacancy_experience",
    "vacancy_employment",
    "vacancy_schedule",
    "vacancy_salary_from",
    "vacancy_salary_to",
    "vacancy_salary_currency",
    "vacancy_salary_gross",
    "vacancy_description",
]

RESUME_COLUMNS = [
    "resume_id",
    "resume_title",
    "resume_specialization",
    "resume_last_position",
    "resume_last_experience_description",
    "resume_last_company_experience_period",
    "resume_skills",
    "resume_education",
    "resume_courses",
    "resume_salary",
    "resume_age",
    "resume_total_experience",
    "resume_experience_months",
    "resume_location",
    "resume_gender",
    "resume_applicant_status",
]

PAIR_COLUMNS = [VACANCY_ID, RESUME_ID, TARGET]

# Fields concatenated into the text that gets embedded (wiki/Векторная_БД.md §4.2).
RESUME_TEXT_FIELDS = [
    "resume_title",
    "resume_specialization",
    "resume_last_position",
    "resume_last_experience_description",
    "resume_skills",
    "resume_education",
    "resume_courses",
]

VACANCY_TEXT_FIELDS = [
    "vacancy_name",
    "vacancy_description",
    "vacancy_experience",
]

# --- Embeddings (Этап 2) ---------------------------------------------------
# Switchable for ablation in eval/. See wiki/Векторная_БД.md.
EMBEDDING_MODEL = "cointegrated/LaBSE-en-ru"
# Alternative: "intfloat/multilingual-e5-large"
EMBEDDING_BATCH_SIZE = 64

# --- Qdrant (Этап 2-3) -----------------------------------------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_RESUMES = "resumes"
COLLECTION_VACANCIES = "vacancies"

# --- Retrieval (Этап 3) ----------------------------------------------------
DEFAULT_TOP_K = 20
# Reciprocal Rank Fusion constant (standard default). See wiki/Hybrid_Search.md.
RRF_K = 60
# How many candidates to pull from each ranker before fusing in hybrid mode.
HYBRID_CANDIDATE_POOL = 100

# --- LLM layer (Этап 5) ----------------------------------------------------
# OpenAI-compatible endpoint. Default: Groq (project uses a Groq API key).
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY_ENV = "GROQ_API_KEY"
# Switchable, like the embedding model. Groq options the project has used:
#   llama-3.3-70b-versatile (default) | openai/gpt-oss-120b | llama-3.1-8b-instant
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2  # low for consistent structured output
PROMPTS_DIR = PROJECT_ROOT / "src" / "llm" / "prompts"

# --- Evaluation (Этап 4) ---------------------------------------------------
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
EVAL_SET_PARQUET = EVAL_RESULTS_DIR / "eval_set.parquet"
EVAL_PERVAC_PARQUET = EVAL_RESULTS_DIR / "per_vacancy_metrics.parquet"
COMPARISON_MD = EVAL_RESULTS_DIR / "comparison.md"
EVAL_K = 10            # report metrics @k
EVAL_RETRIEVE_DEPTH = 100  # depth pulled per query (MRR can see beyond k)
EVAL_SAMPLE_SIZE = 500     # vacancies sampled for the comparison (0 = all)
EVAL_SEED = 42
