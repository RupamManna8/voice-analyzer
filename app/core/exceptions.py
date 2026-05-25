from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class SpeechAPIError(Exception):
    def __init__(self, message: str, *, status_code: int = status.HTTP_400_BAD_REQUEST, error_code: str = "speech_api_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class UnsupportedAudioFormatError(SpeechAPIError):
    def __init__(self, message: str = "Unsupported audio file format") -> None:
        super().__init__(message, status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, error_code="unsupported_audio_format")


class UploadTooLargeError(SpeechAPIError):
    def __init__(self, message: str = "Uploaded file exceeds the maximum allowed size") -> None:
        super().__init__(message, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, error_code="upload_too_large")


class ProcessingError(SpeechAPIError):
    def __init__(self, message: str = "Unable to process the uploaded audio") -> None:
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, error_code="processing_error")


def _error_response(request: Request, status_code: int, error_code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "status": "error",
            "error": {
                "code": error_code,
                "message": message,
            },
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SpeechAPIError)
    async def speech_api_error_handler(request: Request, exc: SpeechAPIError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.error_code, exc.message)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return _error_response(request, exc.status_code, "http_error", detail)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        message = exc.errors()[0].get("msg", "Validation error") if exc.errors() else "Validation error"
        return _error_response(request, status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error", message)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_server_error", "An unexpected error occurred")
