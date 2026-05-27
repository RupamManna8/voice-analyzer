from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import librosa
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioChunkFeatureData:
    start_sec: float
    end_sec: float
    energy: float
    volume_stability: float
    intensity: float
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flatness: float
    spectral_flux: float
    jitter: float
    shimmer: float
    pitch_mean: float
    pitch_median: float
    pitch_variation: float
    relative_pitch_variation: float
    pitch_slope: float
    harmonicity: float


@dataclass(slots=True)
class AudioFeatureData:
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
    
    # Single Source of Truth Sequence Cache
    chunks: list[AudioChunkFeatureData]
    active_speech_waveform: np.ndarray


@dataclass(slots=True)
class PauseAnalysisData:
    total_pauses: int
    long_pauses: int
    average_pause_sec: float
    pause_lengths: list[float]


def load_audio_file(file_path: Path, *, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    waveform, sample_rate = librosa.load(str(file_path), sr=target_sample_rate, mono=True)
    return waveform.astype(np.float32), sample_rate


def read_audio_metadata(file_path: Path) -> tuple[float, int, int]:
    info = sf.info(str(file_path))
    duration_sec = float(info.frames / info.samplerate) if info.samplerate else 0.0
    return duration_sec, int(info.samplerate or 16000), int(info.channels or 1)


def analyze_pause_intervals(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    top_db: float = 30.0,
    min_pause_sec: float = 0.35,
    long_pause_threshold_sec: float = 1.5,
) -> PauseAnalysisData:
    if waveform.size == 0:
        return PauseAnalysisData(total_pauses=0, long_pauses=0, average_pause_sec=0.0, pause_lengths=[])

    non_silent_intervals = librosa.effects.split(waveform, top_db=top_db)
    if len(non_silent_intervals) <= 1:
        return PauseAnalysisData(total_pauses=0, long_pauses=0, average_pause_sec=0.0, pause_lengths=[])

    pause_lengths: list[float] = []
    for index in range(1, len(non_silent_intervals)):
        previous_end = non_silent_intervals[index - 1][1]
        current_start = non_silent_intervals[index][0]
        pause_seconds = max(0.0, (current_start - previous_end) / float(sample_rate))
        if pause_seconds >= min_pause_sec:
            pause_lengths.append(pause_seconds)

    if not pause_lengths:
        return PauseAnalysisData(total_pauses=0, long_pauses=0, average_pause_sec=0.0, pause_lengths=[])

    total_pauses = len(pause_lengths)
    long_pauses = sum(1 for pause in pause_lengths if pause >= long_pause_threshold_sec)
    average_pause_sec = float(np.mean(pause_lengths))

    return PauseAnalysisData(
        total_pauses=total_pauses,
        long_pauses=long_pauses,
        average_pause_sec=round(average_pause_sec, 2),
        pause_lengths=pause_lengths,
    )


def extract_audio_features(file_path: Path) -> AudioFeatureData:
    settings = get_settings()
    duration_sec, sample_rate, channels = read_audio_metadata(file_path)
    
    # 1. Preprocessing: Load mono 16kHz audio
    waveform, sr = load_audio_file(file_path, target_sample_rate=16000)
    
    # Amplitude Peak Normalization
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / (max_amp + 1e-9)

    # 2. Voice Activity Detection (VAD) Speech Extraction
    non_silent_intervals = librosa.effects.split(waveform, top_db=30)
    if non_silent_intervals.size > 0:
        # Concatenate active speech segments only
        active_speech_waveform = np.concatenate([
            waveform[start:end] for start, end in non_silent_intervals
        ])
    else:
        active_speech_waveform = waveform
        
    if active_speech_waveform.size == 0:
        active_speech_waveform = waveform

    # Peak normalization on active speech segment
    active_max = np.max(np.abs(active_speech_waveform))
    if active_max > 0:
        active_speech_waveform = active_speech_waveform / (active_max + 1e-9)

    # Handle fully empty/silent audio files
    if waveform.size == 0 or active_speech_waveform.size == 0:
        return AudioFeatureData(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            energy=0.0,
            pitch_variation=0.0,
            volume_stability=0.0,
            noise_score=1.0,
            zero_crossing_rate=0.0,
            spectral_centroid=0.0,
            spectral_rolloff=0.0,
            jitter=0.0,
            shimmer=0.0,
            intensity=-120.0,
            pitch_mean=0.0,
            relative_pitch_variation=0.0,
            pitch_slope=0.0,
            speech_rate=0.0,
            pause_frequency=0.0,
            pause_duration=0.0,
            spectral_flatness=0.0,
            spectral_flux=0.0,
            mfccs=[0.0] * 13,
            harmonicity=0.0,
            speaking_tempo=0.0,
            pause_irregularity=0.0,
            articulation_continuity=0.0,
            chunks=[],
            active_speech_waveform=np.array([], dtype=np.float32),
        )

    # 3. Global Pitch Extraction (librosa.pyin)
    # Ignores unvoiced frames, removes NaNs, and provides relative pitch variance
    f0, voiced_flag, voiced_probs = librosa.pyin(
        active_speech_waveform,
        sr=16000,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    valid_f0 = f0[~np.isnan(f0)]
    
    if valid_f0.size:
        pitch_mean = float(np.mean(valid_f0))
        pitch_median = float(np.median(valid_f0))
        pitch_variation = float(np.std(valid_f0))
        # Relative pitch variation: std / median
        relative_pitch_variation = float(np.std(valid_f0) / (pitch_median + 1e-6))
        
        if valid_f0.size > 1:
            x = np.arange(valid_f0.size)
            slope, _ = np.polyfit(x, valid_f0, 1)
            pitch_slope = float(slope)
        else:
            pitch_slope = 0.0
    else:
        pitch_mean = 0.0
        pitch_median = 0.0
        pitch_variation = 0.0
        relative_pitch_variation = 0.0
        pitch_slope = 0.0

    # 4. Global Parselmouth Praat Perturbations (Jitter & Shimmer)
    # Extracted exactly once from the active speech VAD wave
    try:
        temp_dir = Path(settings.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save VAD speech to temporary WAV file to invoke Praat correlation
        from uuid import uuid4
        temp_active_path = temp_dir / f"temp_active_{uuid4().hex}.wav"
        sf.write(str(temp_active_path), active_speech_waveform, 16000)
        
        sound = parselmouth.Sound(str(temp_active_path))
        pitch_obj = sound.to_pitch()
        pulses = call([sound, pitch_obj], "To PointProcess (cc)")
        
        # Micro-perturbation local jitter and local shimmer
        jitter_val = call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        shimmer_val = call([sound, pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        
        # Harmonicity HNR
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 4.5)
        hnr_val = call(harmonicity, "Get mean", 0.0, 0.0)
        
        jitter = float(jitter_val) if not np.isnan(jitter_val) else 0.0
        shimmer = float(shimmer_val) if not np.isnan(shimmer_val) else 0.0
        harmonicity_val = float(hnr_val) if not np.isnan(hnr_val) else 0.0
        
        # Noise-Compensated Vocal Perturbation Estimator
        # Low HNR (high background noise) mathematically inflates jitter/shimmer.
        # We apply an HNR-calibrated correction factor to scale them to true clean bounds.
        if harmonicity_val < 15.0:
            factor = max(0.15, min(1.0, harmonicity_val / 15.0))
            jitter = jitter * factor
            shimmer = shimmer * factor
        
        if temp_active_path.exists():
            temp_active_path.unlink()
    except Exception as e:
        logger.warning("Praat feature extraction failed: %s", e)
        jitter = 0.0
        shimmer = 0.0
        harmonicity_val = 0.0

    # 5. Global Energy, Intensity, Centroid, Flatness, Flux from Active Speech
    rms = librosa.feature.rms(y=active_speech_waveform)[0]
    energy = float(np.mean(rms)) if rms.size else 0.0
    
    # Intensity fix: amplitude_to_db RMS mean
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    intensity = float(np.mean(rms_db))

    # Volume stability fix: 1.0 - std / mean
    rms_std = float(np.std(rms)) if rms.size else 0.0
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    volume_stability = float(1.0 - np.clip(rms_std / (rms_mean + 1e-6), 0.0, 1.0)) if rms.size else 1.0

    # Noise score fix: spectral flatness mean
    flatness = librosa.feature.spectral_flatness(y=active_speech_waveform)[0]
    noise_score = float(np.mean(flatness)) if flatness.size else 0.0
    spectral_flatness = noise_score

    # Spectral Centroid & Rolloff
    centroid = librosa.feature.spectral_centroid(y=active_speech_waveform, sr=16000)[0]
    spectral_centroid = float(np.mean(centroid)) if centroid.size else 0.0
    
    rolloff = librosa.feature.spectral_rolloff(y=active_speech_waveform, sr=16000)[0]
    spectral_rolloff = float(np.mean(rolloff)) if rolloff.size else 0.0

    # Spectral Flux (centroid derivative mean)
    spectral_flux = float(np.mean(np.abs(np.diff(centroid)))) if centroid.size > 1 else 0.0

    # MFCCs
    mfcc_feats = librosa.feature.mfcc(y=active_speech_waveform, sr=16000, n_mfcc=13)
    mfccs = [float(np.mean(c)) for c in mfcc_feats] if mfcc_feats.size else [0.0] * 13

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=active_speech_waveform)[0]
    zero_crossing_rate = float(np.mean(zcr)) if zcr.size else 0.0

    # 6. Global Pacing & pauses from full wave
    pause_analysis = analyze_pause_intervals(
        waveform,
        sample_rate=16000,
        top_db=30.0,
        min_pause_sec=0.35,
        long_pause_threshold_sec=1.5,
    )
    total_pauses = pause_analysis.total_pauses
    avg_pause_sec = pause_analysis.average_pause_sec
    pause_lengths = pause_analysis.pause_lengths

    speech_duration = float(active_speech_waveform.size / 16000.0)
    speaking_tempo = float(speech_duration / (duration_sec + 1e-6))
    pause_frequency = float(total_pauses / (duration_sec + 1e-6))
    
    if len(pause_lengths) > 1:
        pause_irregularity = float(np.std(pause_lengths))
    else:
        pause_irregularity = 0.0

    try:
        voiced_duration = len(valid_f0) * 0.01
        articulation_continuity = float(voiced_duration / (duration_sec + 1e-6))
    except Exception:
        articulation_continuity = 0.0

    # Speech rate placeholder (scoring service overrides with total words)
    speech_rate = 0.0

    # 7. Pre-compute sliding 3-second temporal chunks (with 1.5-second overlaps)
    # Extracts the shared feature vector sequence strictly once
    chunks: list[AudioChunkFeatureData] = []
    chunk_samples = int(3.0 * 16000)
    step_samples = int(1.5 * 16000)
    num_samples = len(active_speech_waveform)
    
    if num_samples > chunk_samples:
        for start_idx in range(0, num_samples - chunk_samples, step_samples):
            end_idx = start_idx + chunk_samples
            chunk_wave = active_speech_waveform[start_idx:end_idx]
            
            start_sec = round(start_idx / 16000.0, 2)
            end_sec = round(end_idx / 16000.0, 2)
            
            # Chunk Pitch (pyin)
            c_f0, c_voiced_flag, c_voiced_probs = librosa.pyin(
                chunk_wave,
                sr=16000,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
            )
            c_valid_f0 = c_f0[~np.isnan(c_f0)]
            
            if c_valid_f0.size:
                c_pitch_mean = float(np.mean(c_valid_f0))
                c_pitch_median = float(np.median(c_valid_f0))
                c_pitch_variation = float(np.std(c_valid_f0))
                c_relative_pitch_variation = float(np.std(c_valid_f0) / (c_pitch_median + 1e-6))
                
                if c_valid_f0.size > 1:
                    x = np.arange(c_valid_f0.size)
                    c_slope, _ = np.polyfit(x, c_valid_f0, 1)
                    c_pitch_slope = float(c_slope)
                else:
                    c_pitch_slope = 0.0
            else:
                c_pitch_mean = 0.0
                c_pitch_median = 0.0
                c_pitch_variation = 0.0
                c_relative_pitch_variation = 0.0
                c_pitch_slope = 0.0
                
            # Chunk Perturbations (Parselmouth)
            try:
                temp_chunk_path = temp_dir / f"temp_chunk_{uuid4().hex}.wav"
                sf.write(str(temp_chunk_path), chunk_wave, 16000)
                
                c_sound = parselmouth.Sound(str(temp_chunk_path))
                c_pitch_obj = c_sound.to_pitch()
                c_pulses = call([c_sound, c_pitch_obj], "To PointProcess (cc)")
                
                c_jitter_val = call(c_pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
                c_shimmer_val = call([c_sound, c_pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
                c_harmonicity = call(c_sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 4.5)
                c_hnr_val = call(c_harmonicity, "Get mean", 0.0, 0.0)
                
                c_jitter = float(c_jitter_val) if not np.isnan(c_jitter_val) else 0.0
                c_shimmer = float(c_shimmer_val) if not np.isnan(c_shimmer_val) else 0.0
                c_hnr = float(c_hnr_val) if not np.isnan(c_hnr_val) else 0.0
                
                # Noise-Compensated Chunk Perturbations
                if c_hnr < 15.0:
                    c_factor = max(0.15, min(1.0, c_hnr / 15.0))
                    c_jitter = c_jitter * c_factor
                    c_shimmer = c_shimmer * c_factor
                
                if temp_chunk_path.exists():
                    temp_chunk_path.unlink()
            except Exception:
                c_jitter = 0.0
                c_shimmer = 0.0
                c_hnr = 0.0
                
            # Chunk Energy and spectral features
            c_rms = librosa.feature.rms(y=chunk_wave)[0]
            c_energy = float(np.mean(c_rms)) if c_rms.size else 0.0
            c_volume_stability = float(1.0 - np.clip(np.std(c_rms) / (c_energy + 1e-6), 0.0, 1.0)) if c_rms.size else 1.0
            
            c_rms_db = librosa.amplitude_to_db(c_rms, ref=np.max)
            c_intensity = float(np.mean(c_rms_db))
            
            c_centroid = librosa.feature.spectral_centroid(y=chunk_wave, sr=16000)[0]
            c_spectral_centroid = float(np.mean(c_centroid)) if c_centroid.size else 0.0
            
            c_rolloff = librosa.feature.spectral_rolloff(y=chunk_wave, sr=16000)[0]
            c_spectral_rolloff = float(np.mean(c_rolloff)) if c_rolloff.size else 0.0
            
            c_flatness = librosa.feature.spectral_flatness(y=chunk_wave)[0]
            c_spectral_flatness = float(np.mean(c_flatness)) if c_flatness.size else 0.0
            
            c_spectral_flux = float(np.mean(np.abs(np.diff(c_centroid)))) if c_centroid.size > 1 else 0.0
            
            chunks.append(
                AudioChunkFeatureData(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    energy=c_energy,
                    volume_stability=c_volume_stability,
                    intensity=c_intensity,
                    spectral_centroid=c_spectral_centroid,
                    spectral_rolloff=c_spectral_rolloff,
                    spectral_flatness=c_spectral_flatness,
                    spectral_flux=c_spectral_flux,
                    jitter=c_jitter,
                    shimmer=c_shimmer,
                    pitch_mean=c_pitch_mean,
                    pitch_median=c_pitch_median,
                    pitch_variation=c_pitch_variation,
                    relative_pitch_variation=c_relative_pitch_variation,
                    pitch_slope=c_pitch_slope,
                    harmonicity=c_hnr,
                )
            )

    # Fallback to single overall chunk if duration is under 3 seconds
    if not chunks:
        chunks.append(
            AudioChunkFeatureData(
                start_sec=0.0,
                end_sec=round(duration_sec, 2),
                energy=energy,
                volume_stability=volume_stability,
                intensity=intensity,
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff,
                spectral_flatness=spectral_flatness,
                spectral_flux=spectral_flux,
                jitter=jitter,
                shimmer=shimmer,
                pitch_mean=pitch_mean,
                pitch_median=pitch_median,
                pitch_variation=pitch_variation,
                relative_pitch_variation=relative_pitch_variation,
                pitch_slope=pitch_slope,
                harmonicity=harmonicity_val,
            )
        )

    return AudioFeatureData(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=channels,
        energy=energy,
        pitch_variation=pitch_variation,
        volume_stability=volume_stability,
        noise_score=noise_score,
        zero_crossing_rate=zero_crossing_rate,
        spectral_centroid=spectral_centroid,
        spectral_rolloff=spectral_rolloff,
        jitter=jitter,
        shimmer=shimmer,
        intensity=intensity,
        pitch_mean=pitch_mean,
        relative_pitch_variation=relative_pitch_variation,
        pitch_slope=pitch_slope,
        speech_rate=speech_rate,
        pause_frequency=pause_frequency,
        pause_duration=avg_pause_sec,
        spectral_flatness=spectral_flatness,
        spectral_flux=spectral_flux,
        mfccs=mfccs,
        harmonicity=harmonicity_val,
        speaking_tempo=speaking_tempo,
        pause_irregularity=pause_irregularity,
        articulation_continuity=articulation_continuity,
        chunks=chunks,
        active_speech_waveform=active_speech_waveform,
    )
