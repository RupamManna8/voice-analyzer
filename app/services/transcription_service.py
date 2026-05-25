from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from app.core.constants import normalize_language_code
from app.core.config import get_settings


@dataclass(slots=True)
class TranscriptionSegmentData:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegmentData]
    language: str


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    settings = get_settings()
    return WhisperModel(
        settings.whisper_model_size,
        device="cpu",
        compute_type=settings.whisper_compute_type,
    )


class TranscriptionService:
    def __init__(self) -> None:
        self._settings = get_settings()

    def transcribe(self, file_path: Path, *, language: str | None = None) -> TranscriptionResult:
        model = get_whisper_model()
        requested_language = normalize_language_code(language)
        segments_iter, info = model.transcribe(
            str(file_path),
            language=requested_language or None,
            beam_size=1,
            vad_filter=True,
        )

        segments: list[TranscriptionSegmentData] = []
        text_parts: list[str] = []

        for segment in segments_iter:
            segment_text = segment.text.strip()
            if segment_text:
                text_parts.append(segment_text)

            segments.append(
                TranscriptionSegmentData(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment_text,
                )
            )

        transcript_text = " ".join(text_parts).strip()
        detected_language = requested_language or normalize_language_code(getattr(info, "language", None)) or self._settings.default_language

        return TranscriptionResult(
            text=transcript_text,
            segments=segments,
            language=str(detected_language),
        )
