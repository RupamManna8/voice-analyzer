from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.exceptions import ProcessingError, SpeechAPIError
from app.schemas.request_schema import AnalyzeAudioRequest
from app.schemas.response_schema import (
    AnalysisMetadataResponse,
    AnalysisResponse,
    BehavioralInsightsResponse,
    CommunicationAnalysisResponse,
    FillerWordMetrics,
    PauseAnalysis,
    SentimentAnalysisResponse,
    SpeechMetricsResponse,
    TranscriptResponse,
    TranscriptSegment,
    VoiceMetricsResponse,
    VocalEmotionResponse,
)
from app.services.audio_feature_service import AudioFeatureResult, AudioFeatureService
from app.services.insights_service import InsightsService
from app.services.scoring_service import ScoringService, SpeechMetricsResult
from app.services.sentiment_service import SentimentResult, SentimentService
from app.services.transcription_service import TranscriptionResult, TranscriptionService
from app.services.emotion_service import EmotionService
from app.utils.file_utils import (
    build_temp_audio_path,
    safe_remove_file,
    save_upload_file,
    validate_audio_upload,
    get_file_extension,
    convert_audio_to_wav,
)
from app.core.constants import ALLOWED_AUDIO_EXTENSIONS


class AnalysisService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._transcription_service = TranscriptionService()
        self._audio_feature_service = AudioFeatureService()
        self._sentiment_service = SentimentService()
        self._scoring_service = ScoringService()
        self._insights_service = InsightsService()
        self._emotion_service = EmotionService()

    async def analyze(self, upload_file: UploadFile, request_data: AnalyzeAudioRequest | None = None) -> AnalysisResponse:
        validate_audio_upload(upload_file)
        request_id = str(uuid4())
        temp_file_path = build_temp_audio_path(self._settings.temp_dir, upload_file.filename, request_id)
        original_filename = upload_file.filename or temp_file_path.name
        language_override = request_data.language if request_data else None
        start_time = time.perf_counter()

        await save_upload_file(upload_file, temp_file_path, max_size_bytes=self._settings.max_upload_size_bytes)

        processed_path = temp_file_path
        converted_path = None

        # If the original filename extension isn't one of the allowed types,
        # attempt to convert the saved file to WAV using ffmpeg.
        extension = get_file_extension(upload_file.filename)
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            try:
                converted_path = temp_file_path.parent / (temp_file_path.stem + '_conv.wav')
                convert_audio_to_wav(temp_file_path, converted_path)
                processed_path = converted_path
            except Exception:
                # Let the centralized handler convert exceptions to API responses
                raise

        try:
            analysis_response = await run_in_threadpool(
                self._analyze_sync,
                processed_path,
                original_filename,
                request_id,
                language_override,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return analysis_response.model_copy(update={"processing_time_ms": elapsed_ms})
        except SpeechAPIError:
            raise
        except Exception as exc:  # pragma: no cover - centralized handler converts to API response
            raise ProcessingError(str(exc)) from exc
        finally:
            # Clean up both the original saved file and any converted file
            safe_remove_file(temp_file_path)
            if converted_path is not None:
                safe_remove_file(converted_path)

    def _analyze_sync(
        self,
        file_path: Path,
        original_filename: str,
        request_id: str,
        language_override: str | None,
    ) -> AnalysisResponse:
        transcription: TranscriptionResult = self._transcription_service.transcribe(file_path, language=language_override)
        audio_features: AudioFeatureResult = self._audio_feature_service.extract(file_path)
        speech_language = transcription.language
        sentiment = self._sentiment_service.analyze(transcription.text, language=speech_language)
        speech_metrics: SpeechMetricsResult = self._scoring_service.build_speech_metrics(
            transcription,
            duration_sec=audio_features.duration_sec,
            audio_features=audio_features,
            language=speech_language,
        )
        communication_analysis = self._scoring_service.build_communication_analysis(
            speech_metrics=speech_metrics,
            voice_metrics=audio_features,
            sentiment=sentiment,
            language=speech_language,
        )
        behavioral_insights = self._insights_service.generate_insights(
            speech_metrics=speech_metrics,
            voice_metrics=audio_features,
            communication_analysis=communication_analysis,
            sentiment=sentiment,
            language=speech_language,
        )
        vocal_emotion = self._emotion_service.analyze_vocal_emotion(
            file_path=file_path,
            voice_metrics=audio_features,
            sentiment=sentiment,
        )

        return AnalysisResponse(
            request_id=request_id,
            status="success",
            processing_time_ms=0,
            metadata=AnalysisMetadataResponse(
                filename=original_filename,
                duration_sec=round(audio_features.duration_sec, 2),
                sample_rate=audio_features.sample_rate,
                channels=audio_features.channels,
                language=transcription.language,
            ),
            transcript=TranscriptResponse(
                text=transcription.text,
                segments=[
                    TranscriptSegment(start=segment.start, end=segment.end, text=segment.text)
                    for segment in transcription.segments
                ],
            ),
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
            sentiment_analysis=SentimentAnalysisResponse(
                overall_sentiment=sentiment.overall_sentiment,
                sentiment_score=round(sentiment.sentiment_score, 2),
            ),
            communication_analysis=CommunicationAnalysisResponse(
                clarity_score=communication_analysis.clarity_score,
                fluency_score=communication_analysis.fluency_score,
                confidence_score=communication_analysis.confidence_score,
                pace_score=communication_analysis.pace_score,
                communication_score=communication_analysis.communication_score,
            ),
            behavioral_insights=BehavioralInsightsResponse(
                strengths=behavioral_insights.strengths,
                issues_detected=behavioral_insights.issues_detected,
                recommendations=behavioral_insights.recommendations,
            ),
            vocal_emotion=vocal_emotion,
        )


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    return AnalysisService()
