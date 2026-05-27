from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
import logging
import re

from app.core.config import get_settings
from app.core.constants import get_language_pack, normalize_language_code


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SentimentResult:
    overall_sentiment: str
    sentiment_score: float


def _normalize_label(label: str, score: float) -> tuple[str, float]:
    normalized = label.strip().lower()
    return normalized, float(score)


@lru_cache(maxsize=1)
def _get_sentiment_pipeline() -> object | None:
    settings = get_settings()
    if settings.disable_transformers:
        logger.warning("Transformers are disabled by configuration; using deterministic sentiment scoring")
        return None
    try:
        transformers_module = import_module("transformers")
    except ImportError:  # pragma: no cover
        logger.warning("Transformers is not installed; using deterministic sentiment scoring")
        return None

    pipeline_factory = getattr(transformers_module, "pipeline", None)
    if pipeline_factory is None:
        logger.warning("Transformers pipeline is unavailable; using deterministic sentiment scoring")
        return None

    try:
        # Load multiclass text emotion classifier
        return pipeline_factory("text-classification", model=settings.sentiment_model_name, device=-1)
    except Exception as exc:  # pragma: no cover
        logger.warning("Falling back to deterministic sentiment scoring: %s", exc)
        return None


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def _fallback_sentiment(text: str, language: str) -> SentimentResult:
    language_pack = get_language_pack(language)
    positive_words = set(language_pack["positive_words"])
    negative_words = set(language_pack["negative_words"])

    tokens = _tokenize_text(text)
    if not tokens:
        return SentimentResult(overall_sentiment="neutral", sentiment_score=0.5)

    positive_hits = sum(1 for token in tokens if token in positive_words)
    negative_hits = sum(1 for token in tokens if token in negative_words)
    total_hits = positive_hits + negative_hits

    if total_hits == 0:
        return SentimentResult(overall_sentiment="neutral", sentiment_score=0.5)

    polarity = (positive_hits - negative_hits) / total_hits
    score = max(0.0, min(1.0, 0.5 + polarity / 2.0))

    if polarity > 0.1:
        return SentimentResult(overall_sentiment="joy", sentiment_score=score)
    if polarity < -0.1:
        return SentimentResult(overall_sentiment="sadness", sentiment_score=1.0 - score)
    return SentimentResult(overall_sentiment="neutral", sentiment_score=0.5)


class SentimentService:
    def analyze(self, text: str, *, language: str | None = None) -> SentimentResult:
        cleaned_text = text.strip()
        if not cleaned_text:
            return SentimentResult(overall_sentiment="neutral", sentiment_score=0.5)

        normalized_language = normalize_language_code(language) or "en"
        if normalized_language == "hi":
            return _fallback_sentiment(cleaned_text, normalized_language)

        analyzer = _get_sentiment_pipeline()
        if analyzer is None:
            return _fallback_sentiment(cleaned_text, normalized_language)

        try:
            result = analyzer(cleaned_text[:4000])[0]
            label, score = _normalize_label(str(result.get("label", "neutral")), float(result.get("score", 0.5)))
            
            # Map low-confidence emotions to neutral
            if score < 0.40:
                return SentimentResult(overall_sentiment="neutral", sentiment_score=score)
                
            return SentimentResult(overall_sentiment=label, sentiment_score=max(0.0, min(1.0, score)))
        except Exception as exc:  # pragma: no cover
            logger.warning("Transformer sentiment failed, using fallback scoring: %s", exc)
            return _fallback_sentiment(cleaned_text, normalized_language)
