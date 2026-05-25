from __future__ import annotations

from dataclasses import dataclass
import re

from app.core.config import get_settings
from app.core.constants import get_language_pack, normalize_language_code
from app.services.audio_feature_service import AudioFeatureResult
from app.services.sentiment_service import SentimentResult
from app.services.transcription_service import TranscriptionResult


@dataclass(slots=True)
class FillerWordBreakdown:
    total: int
    details: dict[str, int]


@dataclass(slots=True)
class PauseAnalysisResult:
    total_pauses: int
    long_pauses: int
    average_pause_sec: float


@dataclass(slots=True)
class SpeechMetricsResult:
    total_words: int
    words_per_minute: int
    filler_words: FillerWordBreakdown
    pause_analysis: PauseAnalysisResult


@dataclass(slots=True)
class CommunicationAnalysisResult:
    clarity_score: float
    fluency_score: float
    confidence_score: float
    pace_score: float
    communication_score: float


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def _count_words(text: str) -> int:
    return len(_tokenize_text(text))


def _count_fillers(text: str, *, language: str) -> FillerWordBreakdown:
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    language_pack = get_language_pack(language)
    filler_words = tuple(language_pack["filler_words"])
    details: dict[str, int] = {}
    remaining_text = lowered
    total = 0

    for filler in sorted(filler_words, key=len, reverse=True):
        pattern = rf"\b{re.escape(filler)}\b"
        matches = list(re.finditer(pattern, remaining_text))
        count = len(matches)
        if count:
            details[filler] = count
            total += count
            remaining_text = re.sub(pattern, " ", remaining_text)

    return FillerWordBreakdown(total=total, details=details)


class ScoringService:
    def build_speech_metrics(
        self,
        transcription: TranscriptionResult,
        *,
        duration_sec: float,
        audio_features: AudioFeatureResult,
        language: str | None = None,
    ) -> SpeechMetricsResult:
        normalized_language = normalize_language_code(language or transcription.language) or "en"
        total_words = _count_words(transcription.text)
        duration_minutes = max(duration_sec / 60.0, 1.0 / 60.0)
        words_per_minute = int(round(total_words / duration_minutes))
        filler_words = _count_fillers(transcription.text, language=normalized_language)
        pause_analysis = PauseAnalysisResult(
            total_pauses=audio_features.pause_analysis.total_pauses,
            long_pauses=audio_features.pause_analysis.long_pauses,
            average_pause_sec=audio_features.pause_analysis.average_pause_sec,
        )

        return SpeechMetricsResult(
            total_words=total_words,
            words_per_minute=words_per_minute,
            filler_words=filler_words,
            pause_analysis=pause_analysis,
        )

    def build_communication_analysis(
        self,
        *,
        speech_metrics: SpeechMetricsResult,
        voice_metrics: AudioFeatureResult,
        sentiment: SentimentResult,
        language: str | None = None,
    ) -> CommunicationAnalysisResult:
        settings = get_settings()
        filler_ratio = speech_metrics.filler_words.total / max(1, speech_metrics.total_words)
        pause_ratio = speech_metrics.pause_analysis.total_pauses / max(1, speech_metrics.total_words / 25)
        long_pause_penalty = speech_metrics.pause_analysis.long_pauses / max(1, speech_metrics.pause_analysis.total_pauses)

        clarity_score = _clamp(
            0.5 * (1.0 - min(1.0, filler_ratio * 3.0))
            + 0.25 * (1.0 - min(1.0, pause_ratio))
            + 0.15 * voice_metrics.volume_stability
            + 0.10 * (1.0 - voice_metrics.noise_score),
        )

        fluency_score = _clamp(
            0.55 * (1.0 - min(1.0, filler_ratio * 4.0))
            + 0.25 * (1.0 - min(1.0, pause_ratio))
            + 0.20 * (1.0 - min(1.0, long_pause_penalty)),
        )

        confidence_score = _clamp(
            0.30 * voice_metrics.energy * 10.0
            + 0.25 * voice_metrics.volume_stability
            + 0.25 * (1.0 - voice_metrics.noise_score)
            + 0.20 * min(1.0, abs(voice_metrics.pitch_variation) / 150.0),
        )

        pace_delta = abs(speech_metrics.words_per_minute - settings.ideal_words_per_minute)
        pace_score = _clamp(1.0 - (pace_delta / max(settings.ideal_words_per_minute, 1)))

        communication_score = _clamp(
            0.30 * clarity_score
            + 0.25 * fluency_score
            + 0.20 * confidence_score
            + 0.15 * pace_score
            + 0.10 * sentiment.sentiment_score,
        )

        return CommunicationAnalysisResult(
            clarity_score=round(clarity_score, 2),
            fluency_score=round(fluency_score, 2),
            confidence_score=round(confidence_score, 2),
            pace_score=round(pace_score, 2),
            communication_score=round(communication_score, 2),
        )
