from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.constants import ALLOWED_AUDIO_EXTENSIONS
from app.core.exceptions import ProcessingError
from app.schemas.request_schema import AnalyzeAudioRequest
from app.schemas.response_schema import EmotionTimelineAPIResponse, SentimentAnalysisResponse
from app.services.analysis_service import AnalysisService, get_analysis_service
from app.services.audio_feature_service import AudioFeatureService
from app.services.emotion_service import EmotionService
from app.services.sentiment_service import SentimentResult
from app.utils.file_utils import (
    build_temp_audio_path,
    convert_audio_to_wav,
    get_file_extension,
    safe_remove_file,
    save_upload_file,
    validate_audio_upload,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Voice Emotion Intelligence"])


@router.post(
    "/api/v1/emotion/timeline",
    response_model=EmotionTimelineAPIResponse,
    summary="Analyze vocal emotion timeline from uploaded file",
)
async def analyze_emotion_timeline(
    file: UploadFile = File(..., description="Audio file in wav, mp3, or m4a format"),
    language: str | None = Form(default=None, description="Optional BCP-47 language code override"),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> EmotionTimelineAPIResponse:
    # 1. Validate audio upload
    validate_audio_upload(file)
    settings = get_settings()
    request_id = str(uuid4())
    temp_file_path = build_temp_audio_path(settings.temp_dir, file.filename, request_id)

    # 2. Save temporary upload file
    await save_upload_file(file, temp_file_path, max_size_bytes=settings.max_upload_size_bytes)

    processed_path = temp_file_path
    converted_path = None

    # 3. Handle non-standard file conversions using ffmpeg
    extension = get_file_extension(file.filename)
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        try:
            converted_path = temp_file_path.parent / (temp_file_path.stem + "_conv.wav")
            convert_audio_to_wav(temp_file_path, converted_path)
            processed_path = converted_path
        except Exception as exc:
            safe_remove_file(temp_file_path)
            raise ProcessingError(f"Failed to convert audio file: {exc}")

    try:
        # Normalize and validate language code if provided
        request_data = AnalyzeAudioRequest(language=language)

        # Run analysis service in threadpool to prevent blocking the event loop
        response = await run_in_threadpool(
            analysis_service._analyze_sync,
            processed_path,
            file.filename,
            request_id,
            request_data.language,
        )

        return EmotionTimelineAPIResponse(
            dominant_emotion=response.vocal_emotion.dominant_emotion,
            confidence=response.vocal_emotion.confidence,
            stress_indicator=response.vocal_emotion.stress_indicator,
            rhythm_insight=response.vocal_emotion.rhythm_insight,
            timeline=response.vocal_emotion.trajectory,
        )
    except Exception as exc:
        raise ProcessingError(str(exc))
    finally:
        # Clean up temporary audio files
        safe_remove_file(temp_file_path)
        if converted_path is not None:
            safe_remove_file(converted_path)


@router.websocket("/ws/emotion-stream")
async def ws_emotion_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Real-time WebSocket emotion stream established")

    settings = get_settings()
    audio_feature_service = AudioFeatureService()
    emotion_service = EmotionService()

    # Pre-configure neutral fallback sentiment
    neutral_sentiment = SentimentResult(overall_sentiment="neutral", sentiment_score=0.5)

    # State variables for stream processing
    pcm_buffer = bytearray()
    processed_bytes_threshold = 96000  # 3 seconds of 16kHz 16-bit PCM mono = 96,000 bytes
    step_bytes = 48000  # 1.5 seconds step size = 48,000 bytes
    processed_count = 0

    # Limit active processing history to last 30 seconds to prevent memory bloat and keep CPU execution sub-50ms
    max_history_seconds = 30
    max_history_bytes = max_history_seconds * 16000 * 2  # 30 * 16000 * 2 = 960,000 bytes

    try:
        while True:
            # Receive binary chunk from client
            data = await websocket.receive_bytes()
            if not data:
                continue

            pcm_buffer.extend(data)

            # Cap buffer size to prevent memory leaks in extremely long sessions
            if len(pcm_buffer) > max_history_bytes:
                # Discard oldest bytes keeping exactly max_history_bytes
                pcm_buffer = pcm_buffer[-max_history_bytes:]
                processed_count = max(0, processed_count - (len(pcm_buffer) - max_history_bytes))

            # Run incremental analysis once we have at least 3 seconds of audio
            # and every 1.5 seconds of newly accumulated audio thereafter
            if len(pcm_buffer) >= processed_bytes_threshold:
                current_len = len(pcm_buffer)
                new_bytes_accumulated = current_len - processed_count

                if processed_count == 0 or new_bytes_accumulated >= step_bytes:
                    # Convert the accumulated PCM bytes to a float32 array
                    pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                    # Write array to a temporary WAV file in the config-specified temp directory
                    temp_chunk_path = Path(settings.temp_dir) / f"ws_stream_{uuid4().hex}.wav"
                    try:
                        # Ensure temp directory exists
                        Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
                        
                        await run_in_threadpool(sf.write, str(temp_chunk_path), pcm_array, 16000)

                        # Extract features and predict emotion using the core optimized single-source engine
                        features = await run_in_threadpool(audio_feature_service.extract, temp_chunk_path)
                        response = await run_in_threadpool(
                            emotion_service.analyze_vocal_emotion,
                            temp_chunk_path,
                            features,
                            neutral_sentiment,
                        )

                        # Send strongly-typed JSON message back to the client
                        payload = {
                            "dominant_emotion": response.dominant_emotion,
                            "confidence": response.confidence,
                            "stress_indicator": response.stress_indicator,
                            "rhythm_insight": response.rhythm_insight,
                            "timeline": response.trajectory,
                        }
                        await websocket.send_text(json.dumps(payload))

                    except Exception as e:
                        logger.warning("Error running WebSocket chunk emotion analysis: %s", e)
                    finally:
                        # Clean up temporary WAV chunk immediately
                        safe_remove_file(temp_chunk_path)

                    processed_count = current_len

            # Small yield to prevent CPU thread starvation
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("WebSocket emotion stream disconnected by client")
    except Exception as e:
        logger.error("Unexpected WebSocket emotion stream error: %s", e)
        try:
            await websocket.close()
        except Exception:
            pass
