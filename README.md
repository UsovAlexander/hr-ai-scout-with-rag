# HR AI Scout — LLM-анализатор соответствия резюме вакансии с RAG

RAG-система: находит релевантные резюме под вакансию через векторный поиск
(Qdrant, hybrid dense+BM25+RRF), объясняет соответствие и gaps через LLM, и
честно измеряет качество retrieval оффлайн (NDCG@k, MRR, Recall@k, Precision@k)
на размеченных `target`-парах.

> 📖 Контекст проекта живёт в **`wiki/`** (паттерн LLM Wiki) — начинать с
> `wiki/Index.md`. Спецификация — `raw/project_spec.md`. Конвенции — `CLAUDE.md`.

## Quickstart

```bash
# 1. Окружение
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Подготовка данных (Этап 1): дедуп сущностей -> parquet
python -m src.data.loader

# 3. Поднять векторную БД и проиндексировать сущности
docker compose up -d qdrant
python -m src.vectorstore.indexer        # Этап 2

# 4. Оффлайн-евалюация retrieval (dense / BM25 / hybrid)
python -m eval.run_eval --limit 500      # Этап 4 -> eval/results/comparison.md

# 5. LLM-слой (нужен Groq-ключ в .env как groq_api_key)
python -m src.llm.pipeline --vacancy_id 126167948 --resume_id 6969174

# 6. UI
streamlit run src/app/streamlit_app.py
```

## Статус

Прогресс по этапам — `wiki/Roadmap.md`. Сделано: Этапы 0–6 (данные, индексация,
retrieval, оффлайн-евалюация, LLM-слой, Streamlit UI). Осталось: Этап 7
(README-документация с метриками и скриншотами).
