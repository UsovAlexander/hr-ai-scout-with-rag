# Hybrid Search

Режим retrieval, комбинирующий dense-поиск и BM25 через RRF. Работает поверх
[[Векторная_БД]], оценивается в [[Евалюация]], встроен в [[Архитектура]].

## Зачем

Чистый dense-поиск на резюме часто проигрывает на **точных совпадениях**
конкретного стека («ClickHouse», «CatBoost»). BM25 ловит лексические
совпадения, dense — семантику. Комбинация даёт лучшее из двух.

## Компоненты

1. **Dense** — эмбеддинговый поиск по коллекции `resumes` (модель из
   [[Векторная_БД]]).
2. **BM25** — `rank_bm25` или Qdrant sparse vectors.
3. **RRF (Reciprocal Rank Fusion)** — объединение рангов. Qdrant Query API
   поддерживает нативно через `prefetch` + `fusion=Fusion.RRF`.

## Фильтры по payload

Помимо семантики/лексики, retrieval поддерживает фильтры по payload-полям
([[Векторная_БД]]): город, опыт, статус поиска. Это реальный сценарий
HR-скрининга, а не просто семантический поиск.

## Интерфейс (Этап 3)

```
search_resumes(vacancy_id, top_k=20, filters=None) -> list[(resume_id, score)]
```

- dense-only — базовый режим;
- hybrid (dense + BM25 + RRF) — альтернативный режим;
- фильтры по payload опциональны.

## Три режима для сравнения

В [[Евалюация]] сравниваются **dense-only / BM25-only / hybrid+RRF** —
таблица в `eval/results/comparison.md`. Это ключевой эксперимент проекта.

## Реализовано (Этап 3)

Код: `src/vectorstore/search.py` — класс `ResumeSearcher` + функция-обёртка
`search_resumes(vacancy_id, top_k=20, filters=None, mode="dense")`.

- **Выбор BM25:** `rank_bm25` (`BM25Okapi`) in-memory — НЕ Qdrant sparse vectors.
  Индекс по корпусу резюме строится один раз и кешируется в
  `dataset/bm25_resumes.pkl` (~35 МБ, gitignored).
- **Запрос BM25:** текст вакансии (`vacancy_name + vacancy_description +
  vacancy_experience`, как для эмбеддинга) токенизируется и подаётся в `get_scores`.
- **Токенизация:** lowercase + `\w+` (unicode). Без лемматизации/стоп-слов
  (см. Open questions).
- **RRF:** собственная реализация, `score(d) = Σ 1/(k + rank_d)`, **k = 60**
  (`config.RRF_K`), ранги с 1. Для hybrid из каждого ранкера берётся пул
  `max(top_k, 100)` кандидатов, затем фьюзинг и срез до `top_k`.
- **Фильтры:** dict `{resume_area, resume_applicant_status,
  min_experience_months, max_age}` → транслируется в Qdrant `Filter`. Для всех
  режимов множество допустимых `resume_id` резолвится **один раз** через Qdrant
  (`scroll`), поэтому dense/bm25/hybrid видят одинаковый кандидат-сет.
- **Точки сравнения:** на вакансии «Разработчик SAP ABAP» hybrid убрал
  PHP-ложные срабатывания, которые выдавал чистый dense — наглядный выигрыш RRF.

CLI: `python -m src.vectorstore.search --vacancy_id <id> --mode hybrid --top_k 5 [--area Москва]`.

## Open questions

- **Qdrant-native hybrid не реализован.** Спека §4.4 хвалит native `prefetch` +
  `Fusion.RRF` через sparse-векторы, но это требует пересоздания коллекции
  `resumes` со sparse-конфигом и реиндексации ([[Векторная_БД]]). Выбран
  in-memory `rank_bm25` — возможный апгрейд на будущее.
- Веса dense vs bm25 в RRF не введены (чистый RRF без весов); при желании
  добавить взвешивание.
- Токенизация наивная: без лемматизации/стоп-слов для русского — потенциальная
  точка улучшения BM25-качества.
- `top_k=20` по умолчанию vs пул 100 для фьюзинга — пул зафиксирован, но не
  тюнился; см. влияние на recall в [[Евалюация]].
