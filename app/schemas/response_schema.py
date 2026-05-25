from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class TranscriptResponse(BaseModel):
    text: str
    segments: list[TranscriptSegment]


class FillerWordMetrics(BaseModel):
    total: int = Field(ge=0)
    details: dict[str, int]


class PauseAnalysis(BaseModel):
    total_pauses: int = Field(ge=0)
    long_pauses: int = Field(ge=0)
    average_pause_sec: float = Field(ge=0)


class SpeechMetricsResponse(BaseModel):
    total_words: int = Field(ge=0)
    words_per_minute: int = Field(ge=0)
    filler_words: FillerWordMetrics
    pause_analysis: PauseAnalysis


class VoiceMetricsResponse(BaseModel):
    energy: float = Field(ge=0)
    pitch_variation: float = Field(ge=0)
    volume_stability: float = Field(ge=0, le=1)
    noise_score: float = Field(ge=0, le=1)


class SentimentAnalysisResponse(BaseModel):
    overall_sentiment: str
    sentiment_score: float = Field(ge=0, le=1)


class CommunicationAnalysisResponse(BaseModel):
    clarity_score: float = Field(ge=0, le=1)
    fluency_score: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    pace_score: float = Field(ge=0, le=1)
    communication_score: float = Field(ge=0, le=1)


class BehavioralInsightsResponse(BaseModel):
    strengths: list[str]
    issues_detected: list[str]
    recommendations: list[str]


class AnalysisMetadataResponse(BaseModel):
    filename: str
    duration_sec: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    language: str


class AnalysisResponse(BaseModel):
    request_id: str
    status: Literal["success"]
    processing_time_ms: int = Field(ge=0)
    metadata: AnalysisMetadataResponse
    transcript: TranscriptResponse
    speech_metrics: SpeechMetricsResponse
    voice_metrics: VoiceMetricsResponse
    sentiment_analysis: SentimentAnalysisResponse
    communication_analysis: CommunicationAnalysisResponse
    behavioral_insights: BehavioralInsightsResponse


class ErrorDetailsResponse(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    status: Literal["error"]
    error: ErrorDetailsResponse
