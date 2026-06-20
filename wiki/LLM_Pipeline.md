# LLM Pipeline

LLM-слой работает **поверх top-K кандидатов** от retrieval ([[Hybrid_Search]])
и выдаёт объяснимый вердикт по кандидату. Место в системе — см. [[Архитектура]].

## Цепочка из трёх промптов (Этап 5)

1. **Извлечение фактов** — нормализация `resume_skills` / experience в
   структурированный профиль (pydantic + `instructor`).
2. **Gap-анализ** — сопоставление профиля с `vacancy_description`, явный
   chain-of-thought: совпадения → gaps (отсутствующие требования) →
   критичность.
3. **Скоринг + объяснение** — финальный `score` 0–100 и 2–3 предложения,
   почему кандидат подходит/не подходит на собеседование.

## Где живут промпты

Промпты — это версионируемые `.txt`-шаблоны, не хардкод в Python:

```
src/llm/prompts/
├── gap_analysis.txt
├── scoring.txt
└── explanation.txt
src/llm/schemas.py     # pydantic-модели structured output
src/llm/pipeline.py    # оркестрация промптов
```

Версионируемость промптов — сигнал зрелости в портфолио.

## Стек LLM

- LLM через OpenAI-совместимый API. **Используется Groq** (endpoint
  `https://api.groq.com/openai/v1`), дефолтная модель
  **`llama-3.3-70b-versatile`**; альтернативы (переключаются через
  `config.LLM_MODEL`): `openai/gpt-oss-120b`, `llama-3.1-8b-instant`.
- `instructor` (Mode.JSON) для structured output по pydantic-схемам.
- Ключ берётся из `GROQ_API_KEY`/`groq_api_key` (env или `.env`, gitignored).

## Связь с остальной системой

- **Вход:** top-K `(resume_id, score)` из [[Hybrid_Search]] + тексты резюме и
  `vacancy_description` из [[Датасет]].
- **Выход:** структурированный профиль, список gaps, score 0–100, объяснение —
  отображается в Streamlit UI ([[Roadmap]] Этап 6).

## Реализовано (Этап 5)

Код: `src/llm/schemas.py` (pydantic), `src/llm/prompts/*.txt` (шаблоны),
`src/llm/pipeline.py` (оркестрация). Запуск:
`python -m src.llm.pipeline --vacancy_id <id> --resume_id <id>`.

- **Три шага → три промпта:** `extraction.txt` → `gap_analysis.txt` →
  `scoring.txt`. Объяснение (`explanation`) вошло в шаг скоринга как поле
  схемы (а не отдельный `explanation.txt` из дерева спеки).
- **Схемы:** `CandidateProfile` → `GapAnalysis` (`Gap.severity ∈
  {critical, minor}` + `reasoning`-CoT) → `MatchVerdict` (`score` 0–100,
  `recommendation ∈ {invite, consider, reject}`, `explanation`). Полный
  результат — `CandidateEvaluation`.
- **Рендеринг промптов** через `{{token}}`-плейсхолдеры (replace, не
  `str.format` — текст резюме может содержать `{`).
- **Живой тест пройден:** пара вакансия 126167948 (SAP ABAP) × резюме →
  валидный профиль, gap-анализ с severity, score 40/`reject` с объяснением.

## Open questions

- ~~Соответствие «шаг ↔ файл промпта»~~ — **решено (Этап 5):**
  `extraction → gap_analysis → scoring`, explanation внутри scoring.
- **Score vs `target`:** на тесте релевантному (по retrieval) резюме LLM дал
  `reject` — LLM строго проверяет перечисленные требования, а `target=1` = факт
  шортлиста. Это разные сигналы; калибровать/валидировать score по `target` —
  открыто (связано с [[Евалюация]]).
- **`years_experience` вышел `null`:** текст для извлечения
  (`build_resume_text`) НЕ включает `resume_experience_months` /
  `resume_total_experience` (они в payload, не в тексте) → модель не вывела стаж.
  Если стаж важен в профиле — добавить эти поля в текст экстракции.
- Текст резюме/вакансии подаётся **полностью** (без усечения под контекст) —
  для длинных описаний может потребоваться обрезка/чанкинг.
