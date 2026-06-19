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

# 3. Поднять векторную БД
docker compose up -d qdrant
```

## Статус

Прогресс по этапам — `wiki/Roadmap.md`. Сделано: Этап 0 (структура), Этап 1
(загрузка + дедуп данных).
