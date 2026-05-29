from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.schemas.response_schema import (
    CommunicationAnalysisAPIResponse,
    SpeechMetricsResponse,
    VoiceMetricsResponse,
    FillerWordMetrics,
    PauseAnalysis,
)
from app.services.audio_feature_service import AudioFeatureService, AudioFeatureResult
from app.services.transcription_service import TranscriptionService, TranscriptionResult
from app.services.sentiment_service import SentimentService, SentimentResult
from app.services.scoring_service import ScoringService, SpeechMetricsResult, CommunicationAnalysisResult
from app.services.insights_service import InsightsService, BehavioralInsightsResult

logger = logging.getLogger(__name__)


class CommunicationService:
    def __init__(self) -> None:
        self._transcription_service = TranscriptionService()
        self._audio_feature_service = AudioFeatureService()
        self._sentiment_service = SentimentService()
        self._scoring_service = ScoringService()
        self._insights_service = InsightsService()

    def analyze(
        self,
        file_path: Path,
        language_override: str | None = None,
    ) -> CommunicationAnalysisAPIResponse:
        logger.info("Starting communication analysis for %s", file_path.name)

        # 1. Reuse existing transcription and feature extraction (single file load)
        transcription: TranscriptionResult = self._transcription_service.transcribe(
            file_path, language=language_override
        )
        audio_features: AudioFeatureResult = self._audio_feature_service.extract(file_path)

        speech_language = transcription.language
        sentiment = self._sentiment_service.analyze(transcription.text, language=speech_language)

        # 2. Reuse speech scoring and insights services
        speech_metrics: SpeechMetricsResult = self._scoring_service.build_speech_metrics(
            transcription,
            duration_sec=audio_features.duration_sec,
            audio_features=audio_features,
            language=speech_language,
        )

        communication_analysis: CommunicationAnalysisResult = self._scoring_service.build_communication_analysis(
            speech_metrics=speech_metrics,
            voice_metrics=audio_features,
            sentiment=sentiment,
            language=speech_language,
        )

        behavioral_insights: BehavioralInsightsResult = self._insights_service.generate_insights(
            speech_metrics=speech_metrics,
            voice_metrics=audio_features,
            communication_analysis=communication_analysis,
            sentiment=sentiment,
            language=speech_language,
        )

        # 3. Dynamic physics-grounded speaking style determination
        wpm = speech_metrics.words_per_minute
        clarity = communication_analysis.clarity_score
        confidence = communication_analysis.confidence_score
        fluency = communication_analysis.fluency_score

        if wpm > 160:
            speaking_style = "Rapid and dynamic"
        elif wpm < 100:
            speaking_style = "Slow and deliberate"
        else:
            if clarity > 0.8 and confidence > 0.8:
                speaking_style = "Clear and polished"
            elif fluency < 0.65:
                speaking_style = "Hesitant and conversational"
            else:
                speaking_style = "Standard conversational"

        return CommunicationAnalysisAPIResponse(
            clarity_score=clarity,
            fluency_score=fluency,
            confidence_score=confidence,
            pace_score=communication_analysis.pace_score,
            communication_score=communication_analysis.communication_score,
            speaking_style=speaking_style,
            strengths=behavioral_insights.strengths,
            issues_detected=behavioral_insights.issues_detected,
            recommendations=behavioral_insights.recommendations,
            speech_metrics=SpeechMetricsResponse(
                total_words=speech_metrics.total_words,
                words_per_minute=speech_metrics.words_per_minute,
                filler_words=FillerWordMetrics(
                    total=speech_metrics.filler_words.total,
                    details=speech_metrics.filler_words.details,
                ),
                pause_analysis=PauseAnalysis(
                    total_pauses=speech_metrics.pause_analysis.total_pauses,
                    long_pauses=speech_metrics.pause_analysis.long_pauses,
                    average_pause_sec=speech_metrics.pause_analysis.average_pause_sec,
                ),
            ),
            voice_metrics=VoiceMetricsResponse(
                energy=round(audio_features.energy, 4),
                pitch_variation=round(audio_features.pitch_variation, 2),
                volume_stability=round(audio_features.volume_stability, 2),
                noise_score=round(audio_features.noise_score, 2),
                jitter=round(audio_features.jitter, 4),
                shimmer=round(audio_features.shimmer, 4),
                spectral_centroid=round(audio_features.spectral_centroid, 2),
                intensity=round(audio_features.intensity, 2),
                harmonicity=round(audio_features.harmonicity, 2),
                spectral_flatness=round(audio_features.spectral_flatness, 4),
            ),
        )


@lru_cache(maxsize=1)
def get_communication_service() -> CommunicationService:
    return CommunicationService()
