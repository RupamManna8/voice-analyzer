from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.schemas.request_schema import AnalyzeAudioRequest
from app.schemas.response_schema import AnalysisResponse
from app.services.analysis_service import AnalysisService, get_analysis_service


router = APIRouter(prefix="/api/v1", tags=["Speech Intelligence"])
compat_router = APIRouter(tags=["Speech Intelligence"])


@router.post("/analyze", response_model=AnalysisResponse, summary="Analyze uploaded speech audio")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file in wav, mp3, or m4a format"),
    language: str | None = Form(default=None, description="Optional BCP-47 language code override"),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    request_data = AnalyzeAudioRequest(language=language)
    return await analysis_service.analyze(file, request_data)


@compat_router.post("/analyze", response_model=AnalysisResponse, include_in_schema=False)
async def analyze_audio_compat(
    file: UploadFile = File(..., description="Audio file in wav, mp3, or m4a format"),
    language: str | None = Form(default=None, description="Optional BCP-47 language code override"),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    request_data = AnalyzeAudioRequest(language=language)
    return await analysis_service.analyze(file, request_data)
