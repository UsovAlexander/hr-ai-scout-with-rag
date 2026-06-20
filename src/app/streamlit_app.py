"""Этап 6 — Streamlit UI for HR AI Scout.

Tab 1 "Поиск кандидатов": pick a vacancy, retrieve top-K resumes (dense / bm25 /
hybrid, with optional payload filters), expand a resume to run the LLM pipeline
(profile -> gap analysis -> verdict).
Tab 2 "Метрики качества": render the offline retrieval comparison from eval/.

Run:  streamlit run src/app/streamlit_app.py
See wiki/Roadmap.md (Этап 6).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `from src...` work when launched via `streamlit run src/app/streamlit_app.py`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config
from src.vectorstore.search import ResumeSearcher

st.set_page_config(page_title="HR AI Scout", page_icon="🧭", layout="wide")

LLM_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
SEVERITY_ICON = {"critical": "🔴", "minor": "🟡"}
RECOMMENDATION_BADGE = {"invite": "🟢 invite", "consider": "🟡 consider", "reject": "🔴 reject"}


# --- cached resources ------------------------------------------------------
@st.cache_resource(show_spinner="Подключение к Qdrant и загрузка BM25...")
def get_searcher() -> ResumeSearcher:
    s = ResumeSearcher()
    s._ensure_bm25()
    return s


@st.cache_data(show_spinner=False)
def load_vacancies() -> pd.DataFrame:
    v = pd.read_parquet(config.VACANCIES_PARQUET)
    v[config.VACANCY_ID] = v[config.VACANCY_ID].astype(str)
    return v


@st.cache_data(show_spinner=False)
def load_resumes_indexed() -> pd.DataFrame:
    r = pd.read_parquet(config.RESUMES_PARQUET)
    r[config.RESUME_ID] = r[config.RESUME_ID].astype(str)
    return r.set_index(config.RESUME_ID)


def run_llm(vacancy_id: str, resume_id: str, model: str) -> dict:
    """Run the LLM pipeline once per (vacancy, resume, model), cached in session."""
    key = (vacancy_id, resume_id, model)
    cache = st.session_state.setdefault("llm_cache", {})
    if key not in cache:
        from src.llm.pipeline import evaluate_candidate

        cache[key] = evaluate_candidate(vacancy_id, resume_id, model).model_dump()
    return cache[key]


# --- search tab ------------------------------------------------------------
def render_search_tab(searcher: ResumeSearcher, vacancies: pd.DataFrame, resumes: pd.DataFrame):
    with st.sidebar:
        st.header("Параметры")
        mode = st.radio("Режим поиска", ["hybrid", "dense", "bm25"], index=0,
                        help="hybrid = dense + BM25 (RRF). Сравнение качества — вкладка «Метрики».")
        top_k = st.slider("Сколько кандидатов (top-K)", 5, 50, config.DEFAULT_TOP_K, step=5)
        model = st.selectbox("LLM-модель (Groq)", LLM_MODELS, index=0)

        with st.expander("Фильтры (опционально)"):
            area = st.text_input("Город (точное совпадение)", "")
            status = st.selectbox("Статус соискателя",
                                  ["любой", "Активно ищет работу", "Рассматривает предложения"])
            min_exp = st.number_input("Мин. опыт, мес.", min_value=0, value=0, step=12)

    filters: dict = {}
    if area.strip():
        filters["resume_area"] = area.strip()
    if status != "любой":
        filters["resume_applicant_status"] = status
    if min_exp > 0:
        filters["min_experience_months"] = int(min_exp)

    # Vacancy picker (searchable by name).
    vacancies = vacancies.copy()
    vacancies["label"] = vacancies["vacancy_name"].fillna("(без названия)") + "  ·  #" + vacancies[config.VACANCY_ID]
    label = st.selectbox("Вакансия", vacancies["label"].tolist(),
                         index=0, placeholder="Начните вводить название...")
    vrow = vacancies.loc[vacancies["label"] == label].iloc[0]
    vacancy_id = vrow[config.VACANCY_ID]

    with st.expander("Описание вакансии", expanded=False):
        st.markdown(f"**{vrow.get('vacancy_name', '')}**  ·  {vrow.get('vacancy_area', '')}  "
                    f"·  опыт: {vrow.get('vacancy_experience', '—')}")
        st.write(vrow.get("vacancy_description", ""))

    st.subheader(f"Top-{top_k} кандидатов · режим `{mode}`"
                 + (f" · фильтры: {filters}" if filters else ""))

    try:
        hits = searcher.search_resumes(vacancy_id, top_k=top_k, filters=filters or None, mode=mode)
    except Exception as exc:  # noqa: BLE001 - surface any retrieval/Qdrant error in UI
        st.error(f"Ошибка поиска: {exc}\n\nЗапущен ли Qdrant (`docker compose up -d qdrant`)?")
        return

    if not hits:
        st.warning("Ничего не найдено под заданные фильтры.")
        return

    for rank, (resume_id, score) in enumerate(hits, start=1):
        rrow = resumes.loc[resume_id] if resume_id in resumes.index else None
        title = rrow["resume_title"] if rrow is not None else "?"
        location = rrow.get("resume_location", "") if rrow is not None else ""
        with st.expander(f"**{rank}. {title}**  ·  score {score:.4f}  ·  {location}  ·  #{resume_id}"):
            if rrow is not None:
                st.caption(
                    f"Специализация: {rrow.get('resume_specialization', '—')} | "
                    f"Последняя должность: {rrow.get('resume_last_position', '—')} | "
                    f"Опыт, мес.: {rrow.get('resume_experience_months', '—')}")
                st.write("**Навыки:** " + str(rrow.get("resume_skills", "—")))
            key = (vacancy_id, resume_id, model)
            shown = st.session_state.setdefault("llm_shown", set())
            if st.button("🤖 LLM-анализ соответствия", key=f"llm-{resume_id}"):
                shown.add(key)
            if key in shown:  # keep result visible across reruns (cached)
                _render_llm_result(vacancy_id, resume_id, model)


def _render_llm_result(vacancy_id: str, resume_id: str, model: str):
    try:
        with st.spinner(f"LLM ({model}): извлечение → gap-анализ → скоринг..."):
            result = run_llm(vacancy_id, resume_id, model)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Ошибка LLM: {exc}\n\nЗадан ли ключ в `.env` (groq_api_key)?")
        return

    verdict, gap, profile = result["verdict"], result["gap_analysis"], result["profile"]

    c1, c2 = st.columns([1, 3])
    c1.metric("Score", f"{verdict['score']}/100")
    c1.markdown(f"### {RECOMMENDATION_BADGE.get(verdict['recommendation'], verdict['recommendation'])}")
    c2.markdown("**Вывод:** " + verdict["explanation"])

    st.markdown("**Сильные стороны:** " + (", ".join(gap["strengths"]) or "—"))
    if gap["gaps"]:
        st.markdown("**Пробелы:**")
        for g in gap["gaps"]:
            st.markdown(f"- {SEVERITY_ICON.get(g['severity'], '')} *{g['severity']}* — {g['requirement']}")
    with st.expander("Рассуждение модели (gap-анализ) и извлечённый профиль"):
        st.markdown("**Reasoning:** " + gap["reasoning"])
        st.json(profile)


# --- metrics tab -----------------------------------------------------------
def render_metrics_tab():
    st.subheader("Качество retrieval (оффлайн-евалюация)")
    if config.COMPARISON_MD.exists():
        st.markdown(config.COMPARISON_MD.read_text(encoding="utf-8"))
    else:
        st.info("Отчёт ещё не сгенерирован. Запустите `python -m eval.run_eval`.")
    if config.EVAL_PERVAC_PARQUET.exists():
        with st.expander("Per-vacancy метрики (таблица)"):
            st.dataframe(pd.read_parquet(config.EVAL_PERVAC_PARQUET), width="stretch")


# --- main ------------------------------------------------------------------
def main():
    st.title("🧭 HR AI Scout")
    st.caption("RAG-поиск релевантных резюме под вакансию + LLM-объяснение соответствия. "
               "Контекст проекта — в `wiki/`.")

    searcher = get_searcher()
    vacancies = load_vacancies()
    resumes = load_resumes_indexed()

    tab_search, tab_metrics = st.tabs(["🔍 Поиск кандидатов", "📊 Метрики качества"])
    with tab_search:
        render_search_tab(searcher, vacancies, resumes)
    with tab_metrics:
        render_metrics_tab()


if __name__ == "__main__":
    main()
else:
    main()
