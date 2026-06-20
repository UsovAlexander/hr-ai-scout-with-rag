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

## Этап 2 — Индексация в Qdrant ✅
- [x] Поднять Qdrant через docker-compose (запущен, v1.18.2).
- [x] Собрать тексты для эмбеддинга (конкатенация полей) — см. [[Векторная_БД]].
- [x] Посчитать эмбеддинги батчами (LaBSE-en-ru, dim 768, `batch_size=64`, MPS).
- [x] Залить коллекции `resumes` (20 845) и `vacancies` (3 409) с payload, Cosine.
- [x] Код: `src/vectorstore/{client.py, indexer.py}`; семантическая проверка пройдена.

> Запуск: `python -m src.vectorstore.indexer`. Известный нюанс: `resume_salary`
> не попал в payload (форматированная строка) — см. [[Векторная_БД]] Open questions.

## Этап 3 — Retrieval ✅
- [x] `search_resumes(vacancy_id, top_k=20, filters=None, mode=...)` — `src/vectorstore/search.py`.
- [x] Фильтры по payload (город, статус поиска, опыт, возраст) через Qdrant `Filter`.
- [x] Три режима: dense / bm25 (`rank_bm25`) / hybrid (собственный RRF, k=60) — см. [[Hybrid_Search]].
- [x] BM25-индекс кешируется (`dataset/bm25_resumes.pkl`); смоук-тест всех режимов + фильтра пройден.

> Hybrid убрал PHP-ложные срабатывания чистого dense на «SAP ABAP». Qdrant-native
> sparse-hybrid НЕ делали (выбран in-memory rank_bm25) — см. [[Hybrid_Search]] Open questions.

## Этап 4 — Оффлайн-евалюация retrieval (ключевая часть) ✅
- [x] `build_eval_set.py` — сэмплинг ground truth из `target=1` (500 вакансий, seed=42).
- [x] `retrieval_metrics.py` — NDCG@10, MRR, Recall@10, Precision@10 с нуля + self-test.
- [x] `run_eval.py` — сравнение random/dense/BM25/hybrid → `eval/results/comparison.md`.
- [ ] (Опц.) сравнить две модели эмбеддингов (e5) — не делали, см. [[Евалюация]].

> **Ключевой вывод:** BM25 ≫ dense (recall@10 0.435 vs 0.112), наивный hybrid не
> обгоняет BM25 (0.243) — мотивирует взвешенный fusion. random совпал с теор. полом.
> Детали и таблица — [[Евалюация]].

## Этап 5 — LLM-слой ✅
- [x] Промпт 1: извлечение фактов (`extraction.txt`, pydantic + `instructor`).
- [x] Промпт 2: gap-анализ с severity + CoT (`gap_analysis.txt`).
- [x] Промпт 3: скоринг 0–100 + рекомендация + объяснение (`scoring.txt`).
- [x] Промпты как `.txt`-шаблоны в `src/llm/prompts/`; код — `src/llm/{schemas,pipeline}.py`.
- [x] Провайдер: Groq (`llama-3.3-70b-versatile`), ключ из `.env`; живой тест пройден.

> Нюанс: релевантному резюме LLM дал `reject` — score vs `target` измеряют разное,
> см. [[LLM_Pipeline]] Open questions.

## Этап 6 — Streamlit-приложение ✅
- [x] Выбор вакансии (searchable selectbox по `vacancy_name` + id) — `src/app/streamlit_app.py`.
- [x] Список top-K резюме с score; режим dense/bm25/hybrid + опц. фильтры (город/статус/опыт) в сайдбаре.
- [x] Разворот резюме → кнопка «LLM-анализ» → профиль + gaps (severity) + score/рекомендация/объяснение.
- [x] Вкладка «Метрики качества» рендерит `eval/results/comparison.md` + per-vacancy таблицу.

> Запуск: `streamlit run src/app/streamlit_app.py`. Проверено через `streamlit.testing`
> (AppTest): скрипт исполняется end-to-end, поиск рендерит кандидатов, без ошибок.
> LLM-вкладка переиспользует протестированный `pipeline.evaluate_candidate` ([[LLM_Pipeline]]).

### Доработки UI (по запросу)
- [x] В карточке резюме показывается **`resume_last_experience_description`** — видно,
  на чём LLM строит выводы.
- [x] **Источник вакансии: «Из базы» / «Своя вакансия»** — можно ввести вакансию
  вручную (или подгрузить текст из базы и отредактировать) и искать под неё.
  Под капотом: `ResumeSearcher.search_by_text` (эмбеддит текст на лету, см.
  [[Hybrid_Search]]) + `pipeline.evaluate_custom_vacancy` ([[LLM_Pipeline]]).

## Этап 7 — README и документация ✅ (кроме скриншота)
- [x] Архитектурная диаграмма (ASCII в `README.md`).
- [x] Таблица метрик retrieval из Этапа 4 + содержательные выводы + ablation.
- [x] Раздел «что нового vs классический classification-подход» (таблица сравнения).
- [x] Полный quickstart, стек, структура проекта, указатель на `wiki/`.
- [ ] GIF/скриншот Streamlit-демо → `docs/demo.png` (нужен живой прогон UI; за пользователем).

## Open questions

- Порядок строгий или этапы 4 и 5 можно вести параллельно? Спека говорит
  «выполнять по порядку», но eval (4) и LLM (5) независимы.
