from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.constants import ALLOWED_AUDIO_EXTENSIONS
from app.core.exceptions import ProcessingError
from app.schemas.request_schema import AnalyzeAudioRequest
from app.schemas.response_schema import CommunicationAnalysisAPIResponse
from app.services.communication_service import CommunicationService, get_communication_service
from app.utils.file_utils import (
    build_temp_audio_path,
    convert_audio_to_wav,
    get_file_extension,
    safe_remove_file,
    save_upload_file,
    validate_audio_upload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/communication", tags=["Communication Intelligence"])


@router.post(
    "/analyze",
    response_model=CommunicationAnalysisAPIResponse,
    summary="Analyze communication and speech quality",
)
async def analyze_communication(
    file: UploadFile = File(..., description="Audio file in wav, mp3, or m4a format"),
    language: str | None = Form(default=None, description="Optional BCP-47 language code override"),
    user_id: str | None = Form(default=None, description="Optional identifier for the user"),
    communication_service: CommunicationService = Depends(get_communication_service),
) -> CommunicationAnalysisAPIResponse:
    # 1. Validate audio upload constraints
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

        # Run analysis service in threadpool to keep FastAPI event loop fully asynchronous and CPU-responsive
        response = await run_in_threadpool(
            communication_service.analyze,
            processed_path,
            request_data.language,
        )
        return response
    except Exception as exc:
        raise ProcessingError(str(exc))
    finally:
        # Clean up all temporal audio files to prevent workspace cluttering
        safe_remove_file(temp_file_path)
        if converted_path is not None:
            safe_remove_file(converted_path)
