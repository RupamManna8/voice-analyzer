from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(slots=True)
class AudioFeatureData:
    duration_sec: float
    sample_rate: int
    channels: int
    energy: float
    pitch_variation: float
    volume_stability: float
    noise_score: float


@dataclass(slots=True)
class PauseAnalysisData:
    total_pauses: int
    long_pauses: int
    average_pause_sec: float


def load_audio_file(file_path: Path, *, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    waveform, sample_rate = librosa.load(str(file_path), sr=target_sample_rate, mono=True)
    return waveform.astype(np.float32), sample_rate


def read_audio_metadata(file_path: Path) -> tuple[float, int, int]:
    info = sf.info(str(file_path))
    duration_sec = float(info.frames / info.samplerate) if info.samplerate else 0.0
    return duration_sec, int(info.samplerate or 16000), int(info.channels or 1)


def extract_audio_features(file_path: Path) -> AudioFeatureData:
    duration_sec, sample_rate, channels = read_audio_metadata(file_path)
    waveform, _ = load_audio_file(file_path, target_sample_rate=16000)

    if waveform.size == 0:
        return AudioFeatureData(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            energy=0.0,
            pitch_variation=0.0,
            volume_stability=0.0,
            noise_score=1.0,
        )

    rms = librosa.feature.rms(y=waveform)[0]
    energy = float(np.mean(rms)) if rms.size else 0.0

    pitches, magnitudes = librosa.piptrack(y=waveform, sr=16000)
    valid_pitches = pitches[magnitudes > np.percentile(magnitudes, 70)]
    pitch_variation = float(np.std(valid_pitches)) if valid_pitches.size else 0.0

    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_std = float(np.std(rms)) if rms.size else 0.0
    volume_stability = float(max(0.0, min(1.0, 1.0 - (rms_std / (rms_mean + 1e-6)))))

    signal_power = float(np.mean(np.square(waveform)))
    residual = waveform - librosa.effects.preemphasis(waveform)
    noise_power = float(np.mean(np.square(residual)))
    snr = 10.0 * np.log10((signal_power + 1e-9) / (noise_power + 1e-9))
    noise_score = float(max(0.0, min(1.0, 1.0 - ((snr + 10.0) / 30.0))))

    return AudioFeatureData(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=channels,
        energy=energy,
        pitch_variation=pitch_variation,
        volume_stability=volume_stability,
        noise_score=noise_score,
    )


def analyze_pause_intervals(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    top_db: float = 30.0,
    min_pause_sec: float = 0.35,
    long_pause_threshold_sec: float = 1.5,
) -> PauseAnalysisData:
    if waveform.size == 0:
        return PauseAnalysisData(total_pauses=0, long_pauses=0, average_pause_sec=0.0)

    non_silent_intervals = librosa.effects.split(waveform, top_db=top_db)
    if len(non_silent_intervals) <= 1:
        return PauseAnalysisData(total_pauses=0, long_pauses=0, average_pause_sec=0.0)

    pause_lengths: list[float] = []
    for index in range(1, len(non_silent_intervals)):
        previous_end = non_silent_intervals[index - 1][1]
        current_start = non_silent_intervals[index][0]
        pause_seconds = max(0.0, (current_start - previous_end) / float(sample_rate))
        if pause_seconds >= min_pause_sec:
            pause_lengths.append(pause_seconds)

    if not pause_lengths:
        return PauseAnalysisData(total_pauses=0, long_pauses=0, average_pause_sec=0.0)

    total_pauses = len(pause_lengths)
    long_pauses = sum(1 for pause in pause_lengths if pause >= long_pause_threshold_sec)
    average_pause_sec = float(np.mean(pause_lengths))

    return PauseAnalysisData(
        total_pauses=total_pauses,
        long_pauses=long_pauses,
        average_pause_sec=round(average_pause_sec, 2),
    )
