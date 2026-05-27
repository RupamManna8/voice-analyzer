from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.constants import ALLOWED_AUDIO_EXTENSIONS, ALLOWED_AUDIO_MIME_TYPES
from app.core.exceptions import UnsupportedAudioFormatError, UploadTooLargeError, ProcessingError
import shutil
import subprocess


def get_file_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_audio_upload(upload_file: UploadFile) -> None:
    # Accept any uploaded file here; conversion to a supported audio format
    # will be attempted in the backend processing pipeline. Size checks
    # are enforced when saving the upload.
    return None


def _ensure_ffmpeg_available() -> None:
    if shutil.which('ffmpeg') is None:
        raise ProcessingError('ffmpeg is required for audio conversion but was not found on PATH')


def convert_audio_to_wav(src: Path, dst: Path) -> None:
    """Convert any audio file to WAV using ffmpeg.

    Raises ProcessingError on failure.
    """
    _ensure_ffmpeg_available()
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use ffmpeg to produce PCM 16-bit WAV mono with a common sample rate
    cmd = [
        'ffmpeg', '-y', '-v', 'error', '-hide_banner', '-i', str(src),
        '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', str(dst)
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise ProcessingError(f'ffmpeg conversion failed: {exc}') from exc


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
