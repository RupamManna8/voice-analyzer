from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.utils.audio_utils import AudioFeatureData, PauseAnalysisData, analyze_pause_intervals, extract_audio_features, load_audio_file


@dataclass(slots=True)
class AudioFeatureResult:
    duration_sec: float
    sample_rate: int
    channels: int
    energy: float
    pitch_variation: float
    volume_stability: float
    noise_score: float
    pause_analysis: PauseAnalysisData


class AudioFeatureService:
    def extract(self, file_path: Path) -> AudioFeatureResult:
        features: AudioFeatureData = extract_audio_features(file_path)
        waveform, sample_rate = load_audio_file(file_path, target_sample_rate=16000)
        pause_analysis = analyze_pause_intervals(
            waveform,
            sample_rate,
            top_db=30.0,
            min_pause_sec=0.35,
            long_pause_threshold_sec=1.5,
        )
        return AudioFeatureResult(
            duration_sec=features.duration_sec,
            sample_rate=features.sample_rate,
            channels=features.channels,
            energy=features.energy,
            pitch_variation=features.pitch_variation,
            volume_stability=features.volume_stability,
            noise_score=features.noise_score,
            pause_analysis=pause_analysis,
        )
