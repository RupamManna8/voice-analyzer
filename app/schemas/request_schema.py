from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.constants import normalize_language_code


class AnalyzeAudioRequest(BaseModel):
    language: str | None = Field(default=None, min_length=2, max_length=10, description="Optional language override, e.g. en")

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = normalize_language_code(value)
        if normalized is None:
            raise ValueError("Supported languages are en and hi")
        return normalized
