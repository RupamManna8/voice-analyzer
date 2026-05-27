from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

from pydantic import BaseModel, Field


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass

_load_dotenv()


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default

    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


class Settings(BaseModel):
    app_name: str = "Speech Intelligence API"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 7500
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    temp_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "temp")
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    max_upload_size_mb: int = 25
    whisper_model_size: str = "base"
    whisper_compute_type: str = "int8"
    default_language: str = "en"
    sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    disable_transformers: bool = False
    ideal_words_per_minute: int = 130
    filler_threshold: int = 5
    long_pause_threshold_sec: float = 1.5
    pause_gap_threshold_sec: float = 0.8

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    temp_dir = Path(os.getenv("TEMP_DIR", str(Path(__file__).resolve().parents[1] / "temp")))

    return Settings(
        app_name=os.getenv("APP_NAME", "Speech Intelligence API"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        api_v1_prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7500")),
        debug=os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"},
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        temp_dir=temp_dir,
        allowed_origins=_split_csv(os.getenv("CORS_ALLOW_ORIGINS"), ["*"]),
        cors_allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").strip().lower() in {"1", "true", "yes", "on"},
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")),
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        default_language=os.getenv("DEFAULT_LANGUAGE", "en"),
        sentiment_model_name=os.getenv(
            "SENTIMENT_MODEL_NAME",
            "distilbert-base-uncased-finetuned-sst-2-english",
        ),
        disable_transformers=os.getenv("DISABLE_TRANSFORMERS", "false").strip().lower() in {"1", "true", "yes", "on"},
        ideal_words_per_minute=int(os.getenv("IDEAL_WORDS_PER_MINUTE", "130")),
        filler_threshold=int(os.getenv("FILLER_THRESHOLD", "5")),
        long_pause_threshold_sec=float(os.getenv("LONG_PAUSE_THRESHOLD_SEC", "1.5")),
        pause_gap_threshold_sec=float(os.getenv("PAUSE_GAP_THRESHOLD_SEC", "0.8")),
    )
