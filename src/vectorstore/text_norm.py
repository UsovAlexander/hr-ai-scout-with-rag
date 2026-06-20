"""Russian-aware text normalization for the BM25 lexical path (Этап 3+).

Tokenize -> lemmatize (pymorphy3) -> drop stopwords. Lemmatization fixes the
morphology problem (разработчик / разработчиком are the same word), and stopword
removal drops uninformative tokens. Used by both the resume corpus index and the
vacancy query so they share one vocabulary. See wiki/Hybrid_Search.md.
"""
from __future__ import annotations

import functools
import re

# Latin + Cyrillic + digit runs (tech terms like "ClickHouse", "1с" survive).
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_SIMPLE_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokens(text: str) -> list[str]:
    """Plain lowercase + \\w+ tokenization (no lemmatization/stopwords)."""
    return _SIMPLE_RE.findall(text.lower())

# Standard NLTK Russian stopword list (hardcoded -> no runtime download).
RUSSIAN_STOPWORDS = frozenset("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда
лучше чуть том нельзя такой им более всегда конечно всю между
""".split())


@functools.lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "pymorphy3 is required for lemmatization: pip install pymorphy3 pymorphy3-dicts-ru"
        ) from exc
    return pymorphy3.MorphAnalyzer()


@functools.lru_cache(maxsize=300_000)
def lemmatize(token: str) -> str:
    """Lemma (normal form) of a single token, cached — tokens repeat a lot."""
    return _morph().parse(token)[0].normal_form


def normalize_tokens(text: str) -> list[str]:
    """Tokenize -> lemmatize -> drop stopwords / 1-char / pure-digit tokens."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 2 or tok.isdigit():
            continue
        lemma = lemmatize(tok)
        if len(lemma) < 2 or lemma in RUSSIAN_STOPWORDS:
            continue
        out.append(lemma)
    return out
