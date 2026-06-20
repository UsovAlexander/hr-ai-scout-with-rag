# 🧭 HR AI Scout — LLM-анализатор соответствия резюме вакансии с RAG

Система, которая по вакансии находит релевантные резюме, **объясняет** через LLM,
почему кандидат подходит или нет, подсвечивает пробелы (gaps) — и, в отличие от
90% похожих pet-проектов, **честно измеряет качество поиска** оффлайн на
размеченных данных.

> 📖 Полный контекст проекта живёт в **[`wiki/`](wiki/Index.md)** (паттерн
> [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)) —
> начинать с `wiki/Index.md`. Спецификация — `raw/project_spec.md`.

## Демо

![HR AI Scout demo](docs/demo.png)

<!-- Скриншот/GIF: запустите `streamlit run src/app/streamlit_app.py` и положите
     картинку в docs/demo.png (выбор вакансии → top-K резюме → LLM-разбор). -->

## Чем это отличается от классического classification-подхода

Исходные данные — размеченные пары `target ∈ {0,1}` (приглашать на собес или нет).
Очевидный путь — обучить классификатор (CatBoost/LightGBM). Этот проект делает
**другое** и добавляет три вещи, которых у классификатора нет:

| | Классический classifier | HR AI Scout (этот проект) |
|---|---|---|
| Что делает | предсказывает `P(target=1)` для **заданной** пары | **находит** кандидатов под вакансию из всех 20 845 резюме |
| Новые кандидаты | рассматривает только размеченные пары | retrieval достаёт релевантных, которых разметка не покрывала |
| Объяснимость | score без обоснования | LLM: профиль → gap-анализ (severity) → score + объяснение |
| Оценка качества | accuracy/ROC-AUC классификации | **retrieval-метрики** (NDCG/MRR/Recall/Precision@k) на ground truth |

`target`-разметка здесь используется не для обучения, а как **golden set** для
честной оффлайн-евалюации качества поиска.

## Архитектура

```
total_df.csv (декартово произведение vacancy × resume, 332 330 пар)
   │
   ├── дедуп вакансий (3 409) ─┐
   └── дедуп резюме   (20 845) ┤   src/data/loader.py
                               ▼
                  build_*_text → эмбеддинги (LaBSE-en-ru, 768d)
                               │   src/vectorstore/indexer.py
                               ▼
                  Qdrant: collection resumes / vacancies (Cosine + payload)
                               │
                               ▼
        Retrieval: search_resumes(vacancy_id, top_k, filters, mode)
        dense | bm25 | hybrid (RRF)        src/vectorstore/search.py
                               │
                               ▼
        LLM-слой: extraction → gap-анализ → scoring (Groq + instructor)
                               │   src/llm/pipeline.py
                               ▼
                       Streamlit UI         src/app/streamlit_app.py

   Параллельно: eval/ — оффлайн-оценка retrieval на target=1/0 парах
   (NDCG@k, MRR, Recall@k, Precision@k), собственная реализация метрик
```

## Ключевые результаты (retrieval, 500 вакансий, k=10)

Ground truth — `target=1` резюме на вакансию; модель эмбеддингов `LaBSE-en-ru`.
Полный отчёт генерируется в [`eval/results/comparison.md`](eval/results/comparison.md).

| mode | recall@10 | precision@10 | MRR | ndcg@10 |
|---|---|---|---|---|
| random | 0.0006 | 0.0004 | 0.0014 | 0.0004 |
| **bm25** | **0.4349** | **0.3128** | **0.7480** | **0.4643** |
| dense | 0.1120 | 0.0772 | 0.2737 | 0.1151 |
| hybrid (RRF) | 0.2427 | 0.1732 | 0.5494 | 0.2677 |

**Содержательные выводы** (а не просто «работает»):
- **BM25 ≫ dense** на этих данных: решение рекрутера сильнее коррелирует с явным
  совпадением навыков/термов (часто латиница: ABAP, SAP, ClickHouse), чем с
  эмбеддинг-близостью по длинному тексту резюме.
- **Наивный равновесный hybrid не обгоняет BM25** — равные веса размывают сильный
  лексический сигнал. Мотивирует взвешенный fusion.
- `random` совпадает с теоретическим полом (~k/|corpus|) — это валидирует
  корректность реализации метрик, а не «случайно хорошие числа».
- **Ablation:** русская лемматизация (pymorphy3) + стоп-слова для BM25 **ухудшили**
  recall@10 (0.435 → 0.348) — оставлены переключателем `config.BM25_LEMMATIZE`
  (default off). Опровергнутая гипотеза задокументирована в
  [`wiki/Hybrid_Search.md`](wiki/Hybrid_Search.md).

> ⚠️ `precision@10` — нижняя оценка: поиск идёт по всем 20 845 резюме, а размечено
> лишь ~98 на вакансию, поэтому непомеченные в топе считаются нерелевантными.
> Надёжнее смотреть на recall@10 / MRR / NDCG. Детали — `wiki/Евалюация.md`.

## Стек

- **Векторная БД:** Qdrant (Docker), две коллекции, payload-фильтры, Cosine
- **Эмбеддинги:** `cointegrated/LaBSE-en-ru` (768d), переключается через config
- **Lexical / hybrid:** `rank_bm25` + собственный RRF (k=60)
- **LLM:** Groq (OpenAI-совместимый API), `llama-3.3-70b-versatile`, `instructor` (structured output)
- **UI:** Streamlit · **Данные:** pandas + parquet · **Eval:** метрики реализованы вручную

## Структура проекта

```
src/
  data/         loader.py (дедуп → parquet), preprocessing.py (текст для эмбеддинга)
  vectorstore/  client.py (эмбеддер+Qdrant), indexer.py, search.py, text_norm.py
  llm/          schemas.py (pydantic), prompts/*.txt, pipeline.py
  app/          streamlit_app.py
eval/           build_eval_set.py, retrieval_metrics.py, run_eval.py, results/
wiki/           LLM Wiki — основной слой знаний о проекте
```

## Quickstart

```bash
# 1. Окружение
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Подготовка данных: дедуп сущностей -> parquet
python -m src.data.loader

# 3. Векторная БД + индексация (LaBSE на GPU/MPS/CPU)
docker compose up -d qdrant
python -m src.vectorstore.indexer

# 4. Оффлайн-евалюация retrieval (dense / BM25 / hybrid)
python -m eval.run_eval --limit 500          # -> eval/results/comparison.md

# 5. LLM-слой (нужен Groq-ключ; положите в .env как groq_api_key='...')
python -m src.llm.pipeline --vacancy_id 126167948 --resume_id 6969174

# 6. UI
streamlit run src/app/streamlit_app.py       # http://localhost:8501
```

Поиск из CLI:
```bash
python -m src.vectorstore.search --vacancy_id 126167948 --mode hybrid --top_k 5
```

## Документация

Проект ведётся по паттерну **LLM Wiki**: `wiki/` — компилированный, взаимосвязанный
слой знаний, который растёт от новых материалов и вопросов. Точка входа —
[`wiki/Index.md`](wiki/Index.md). Прогресс по этапам — [`wiki/Roadmap.md`](wiki/Roadmap.md).
