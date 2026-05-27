from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.utils.audio_utils import AudioFeatureData, AudioChunkFeatureData, PauseAnalysisData, analyze_pause_intervals, extract_audio_features, load_audio_file


@dataclass(slots=True)
class AudioFeatureResult:
    duration_sec: float
    sample_rate: int
    channels: int
    energy: float
    pitch_variation: float
    volume_stability: float
    noise_score: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_rolloff: float
    jitter: float
    shimmer: float
    intensity: float
    pause_analysis: PauseAnalysisData
    
    # New advanced features
    pitch_mean: float
    relative_pitch_variation: float
    pitch_slope: float
    speech_rate: float
    pause_frequency: float
    pause_duration: float
    spectral_flatness: float
    spectral_flux: float
    mfccs: list[float]
    harmonicity: float
    speaking_tempo: float
    pause_irregularity: float
    articulation_continuity: float
    
    # Single Source of Truth sequence cache
    chunks: list[AudioChunkFeatureData]
    active_speech_waveform: np.ndarray


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
            zero_crossing_rate=features.zero_crossing_rate,
            spectral_centroid=features.spectral_centroid,
            spectral_rolloff=features.spectral_rolloff,
            jitter=features.jitter,
            shimmer=features.shimmer,
            intensity=features.intensity,
            pause_analysis=pause_analysis,
            pitch_mean=features.pitch_mean,
            relative_pitch_variation=features.relative_pitch_variation,
            pitch_slope=features.pitch_slope,
            speech_rate=features.speech_rate,
            pause_frequency=features.pause_frequency,
            pause_duration=features.pause_duration,
            spectral_flatness=features.spectral_flatness,
            spectral_flux=features.spectral_flux,
            mfccs=features.mfccs,
            harmonicity=features.harmonicity,
            speaking_tempo=features.speaking_tempo,
            pause_irregularity=features.pause_irregularity,
            articulation_continuity=features.articulation_continuity,
            chunks=features.chunks,
            active_speech_waveform=features.active_speech_waveform,
        )
