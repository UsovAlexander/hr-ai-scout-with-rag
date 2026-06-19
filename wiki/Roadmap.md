# Roadmap

Этапы реализации из `raw/project_spec.md` (§5). Чекбоксы отражают **текущее
состояние** — на момент создания вики код ещё не написан, поэтому всё `[ ]`.
Обновлять по мере реализации.

Концептуальные детали каждого этапа — на тематических страницах: [[Датасет]],
[[Векторная_БД]], [[Hybrid_Search]], [[LLM_Pipeline]], [[Евалюация]],
[[Архитектура]].

## Этап 0 — Окружение и структура проекта ✅
- [x] Создать дерево каталогов (`src/`, `eval/`, `notebooks/`, …) — см. [[Архитектура]].
- [x] `docker-compose.yml` для Qdrant — см. [[Векторная_БД]].
- [x] `requirements.txt`.
- [x] `.gitignore` (исключает 2GB csv, parquet, `qdrant_storage/`, `.venv/`), `README.md`-стаб, `src/config.py`.
- [x] venv `.venv/` + установлены pandas/pyarrow (Этап-1 зависимости).

## Этап 1 — Подготовка данных ✅
- [x] Загрузить `total_df.csv` экономно по памяти (chunksize=100k) — `src/data/loader.py`.
- [x] Дедуп вакансий по `vacancy_id`, резюме по `resume_id` — см. [[Датасет]].
- [x] Сохранить `vacancies.parquet` (3 409), `resumes.parquet` (20 845).
- [x] Сохранить таблицу пар `vacancy_id, resume_id, target` → `pairs.parquet` (332 330; target 25 520 / 306 810) — eval ground truth.
- [x] Сборка текста для эмбеддинга — `src/data/preprocessing.py`.

> Все счётчики после запуска точно совпали со спекой. Запуск: `python -m src.data.loader` (~11 c).

## Этап 2 — Индексация в Qdrant
- [ ] Поднять Qdrant через docker-compose.
- [ ] Собрать тексты для эмбеддинга (конкатенация полей) — см. [[Векторная_БД]].
- [ ] Посчитать эмбеддинги батчами (`sentence-transformers`, `batch_size=64`).
- [ ] Залить коллекции `resumes` и `vacancies` с payload.

## Этап 3 — Retrieval
- [ ] `search_resumes(vacancy_id, top_k=20, filters=None)`.
- [ ] Фильтры по payload (город, опыт, статус поиска).
- [ ] Hybrid-режим (dense + BM25 + RRF) — см. [[Hybrid_Search]].

## Этап 4 — Оффлайн-евалюация retrieval (ключевая часть)
- [ ] `build_eval_set.py` — сэмплинг из `target=1/0`.
- [ ] `retrieval_metrics.py` — NDCG@10, MRR, Recall@10, Precision@10.
- [ ] Сравнить dense / BM25 / hybrid+RRF → `eval/results/comparison.md`.
- [ ] (Опц.) сравнить две модели эмбеддингов — см. [[Евалюация]].

## Этап 5 — LLM-слой
- [ ] Промпт 1: извлечение фактов (pydantic + `instructor`).
- [ ] Промпт 2: gap-анализ (chain-of-thought).
- [ ] Промпт 3: скоринг 0–100 + объяснение.
- [ ] Промпты как `.txt`-шаблоны в `src/llm/prompts/` — см. [[LLM_Pipeline]].

## Этап 6 — Streamlit-приложение
- [ ] Выбор вакансии (dropdown/поиск по `vacancy_name`).
- [ ] Список top-K резюме с score.
- [ ] Клик по резюме → gap-анализ + объяснение.
- [ ] Вкладка «Метрики качества» из `eval/results/`.

## Этап 7 — README и документация
- [ ] Архитектурная диаграмма.
- [ ] Таблица метрик retrieval из Этапа 4.
- [ ] GIF/скриншот Streamlit-демо.
- [ ] Раздел «что нового vs классический classification-подход».

## Open questions

- Порядок строгий или этапы 4 и 5 можно вести параллельно? Спека говорит
  «выполнять по порядку», но eval (4) и LLM (5) независимы.
