from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.constants import ALLOWED_AUDIO_EXTENSIONS, ALLOWED_AUDIO_MIME_TYPES
from app.core.exceptions import UnsupportedAudioFormatError, UploadTooLargeError


def get_file_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_audio_upload(upload_file: UploadFile) -> None:
    extension = get_file_extension(upload_file.filename)
    content_type = (upload_file.content_type or "").lower()

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported file extension '{extension or 'unknown'}'. Allowed: wav, mp3, m4a"
        )

    if content_type and content_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise UnsupportedAudioFormatError(
            f"Unsupported content type '{content_type}'. Allowed audio MIME types only"
        )


def build_temp_audio_path(temp_dir: Path, original_filename: str | None, request_id: str) -> Path:
    extension = get_file_extension(original_filename)
    safe_extension = extension if extension in ALLOWED_AUDIO_EXTENSIONS else ".wav"
    return temp_dir / f"{request_id}_{uuid4().hex}{safe_extension}"


async def save_upload_file(upload_file: UploadFile, destination: Path, *, max_size_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    try:
        with destination.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    raise UploadTooLargeError()

                buffer.write(chunk)
    finally:
        await upload_file.close()


def safe_remove_file(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except OSError:
        pass
