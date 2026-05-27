from __future__ import annotations

import logging
import time
from uuid import uuid4

from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from app.api.routes.analyze import compat_router as analyze_compat_router
from app.api.routes.analyze import router as analyze_router
from app.api.routes.communication import router as communication_router
from app.api.routes.emotion import router as emotion_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    # Dynamic CORS configuration to prevent Starlette runtime error with wildcard origins + allow_credentials
    allowed_origins = list(settings.allowed_origins)
    if settings.cors_allow_credentials and "*" in allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=".*",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)

    @app.on_event("startup")
    def preload_models():
        # Pre-warm ML models during application startup so that user requests are blazingly fast!
        import logging
        logger = logging.getLogger("app.main")
        logger.info("==================================================")
        logger.info("Pre-warming ML models during application startup...")
        
        try:
            from app.services.transcription_service import get_whisper_model
            logger.info("Loading Whisper neural model into memory...")
            get_whisper_model()
            logger.info("Whisper neural model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to pre-warm Whisper model: {e}")
            
        try:
            from app.services.sentiment_service import _get_sentiment_pipeline
            logger.info("Loading Sentiment Transformer model into memory...")
            _get_sentiment_pipeline()
            logger.info("Sentiment Transformer model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to pre-warm Sentiment model: {e}")

        try:
            from app.services.emotion_service import get_ser_pipeline
            logger.info("Loading Wav2Vec2 SER model into memory...")
            get_ser_pipeline()
            logger.info("Wav2Vec2 SER model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to pre-warm Wav2Vec2 SER model: {e}")
            
        logger.info("All ML models are fully warmed up and cached in memory!")
        logger.info("==================================================")

    @app.middleware("http")
    async def request_timing_middleware(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()

        response = await call_next(request)
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = str(processing_time_ms)
        return response

    @app.get("/", summary="AudioModel API Test Frontend", response_model=None)
    async def serve_frontend_or_health() -> HTMLResponse | JSONResponse:
        frontend_path = Path(__file__).resolve().parent / "static" / "test_frontend.html"
        if frontend_path.exists():
            try:
                html_content = frontend_path.read_text(encoding="utf-8")
                return HTMLResponse(content=html_content)
            except Exception as e:
                logging.getLogger("app.main").warning(f"Failed to read frontend file: {e}")
        
        return JSONResponse(
            content={
                "status": "healthy",
                "service": settings.app_name,
                "version": settings.app_version,
            }
        )

    app.include_router(analyze_router)
    app.include_router(analyze_compat_router)
    app.include_router(communication_router)
    app.include_router(emotion_router)
    return app


app = create_app()
