from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.analyze import compat_router as analyze_compat_router
from app.api.routes.analyze import router as analyze_router
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

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

    @app.get("/", summary="Health check")
    async def health_check() -> JSONResponse:
        return JSONResponse(
            content={
                "status": "healthy",
                "service": settings.app_name,
                "version": settings.app_version,
            }
        )

    app.include_router(analyze_router)
    app.include_router(analyze_compat_router)
    return app


app = create_app()
