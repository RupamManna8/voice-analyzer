from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.constants import get_language_pack, normalize_language_code
from app.services.audio_feature_service import AudioFeatureResult
from app.services.scoring_service import CommunicationAnalysisResult, SpeechMetricsResult
from app.services.sentiment_service import SentimentResult


@dataclass(slots=True)
class BehavioralInsightsResult:
    strengths: list[str]
    issues_detected: list[str]
    recommendations: list[str]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _phrase(language: str | None, key: str) -> str:
    language_pack = get_language_pack(language)
    insights = language_pack["insights"]
    return str(insights[key])


class InsightsService:
    def generate_insights(
        self,
        *,
        speech_metrics: SpeechMetricsResult,
        voice_metrics: AudioFeatureResult,
        communication_analysis: CommunicationAnalysisResult,
        sentiment: SentimentResult,
        language: str | None = None,
    ) -> BehavioralInsightsResult:
        settings = get_settings()
        normalized_language = normalize_language_code(language) or "en"
        strengths: list[str] = []
        issues_detected: list[str] = []
        recommendations: list[str] = []

        if communication_analysis.communication_score >= 0.8:
            _append_unique(strengths, _phrase(normalized_language, "strong_overall"))

        if communication_analysis.clarity_score >= 0.75:
            _append_unique(strengths, _phrase(normalized_language, "clear_delivery"))

        if communication_analysis.confidence_score >= 0.75:
            _append_unique(strengths, _phrase(normalized_language, "confident_delivery"))

        if sentiment.overall_sentiment == "positive" and sentiment.sentiment_score >= 0.6:
            _append_unique(strengths, _phrase(normalized_language, "positive_tone"))

        if speech_metrics.filler_words.total > settings.filler_threshold:
            _append_unique(issues_detected, _phrase(normalized_language, "frequent_fillers"))
            _append_unique(recommendations, _phrase(normalized_language, "reduce_fillers"))

        if speech_metrics.pause_analysis.long_pauses > 0:
            _append_unique(issues_detected, _phrase(normalized_language, "extended_pauses"))
            _append_unique(recommendations, _phrase(normalized_language, "shorten_pauses"))

        if voice_metrics.noise_score > 0.35:
            _append_unique(issues_detected, _phrase(normalized_language, "background_noise"))
            _append_unique(recommendations, _phrase(normalized_language, "quieter_environment"))

        if communication_analysis.pace_score < 0.65:
            _append_unique(issues_detected, _phrase(normalized_language, "pace_outside"))
            _append_unique(recommendations, _phrase(normalized_language, "adjust_pace"))

        if communication_analysis.fluency_score < 0.65:
            _append_unique(issues_detected, _phrase(normalized_language, "fluency_improve"))
            _append_unique(recommendations, _phrase(normalized_language, "smoother_transitions"))

        if communication_analysis.confidence_score < 0.65:
            _append_unique(issues_detected, _phrase(normalized_language, "confidence_weak"))
            _append_unique(recommendations, _phrase(normalized_language, "increase_projection"))

        if sentiment.overall_sentiment == "negative":
            _append_unique(issues_detected, _phrase(normalized_language, "negative_tone"))
            _append_unique(recommendations, _phrase(normalized_language, "clearer_positive_language"))

        if not strengths:
            _append_unique(strengths, _phrase(normalized_language, "consistent_audio"))

        return BehavioralInsightsResult(
            strengths=strengths,
            issues_detected=issues_detected,
            recommendations=recommendations,
        )
