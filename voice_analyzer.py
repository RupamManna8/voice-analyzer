# file: advanced_audio_api.py

from fastapi import FastAPI, UploadFile, File
import os, uuid, shutil
import numpy as np
import librosa
import spacy
from faster_whisper import WhisperModel
import uvicorn

# ================= INIT =================
app = FastAPI()

model = WhisperModel("base", compute_type="int8")
nlp = spacy.load("en_core_web_sm")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FILLERS = ["um", "uh", "like", "you know", "basically", "actually"]
IDEAL_WPM = 130

# ================= AUDIO FEATURES =================
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    # Energy (loudness)
    rms = librosa.feature.rms(y=y)[0]
    energy = np.mean(rms)

    # Pitch (confidence proxy)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_var = np.var(pitch_values) if len(pitch_values) > 0 else 0

    # Noise estimation (low SNR → noisy)
    signal_power = np.mean(y**2)
    noise_power = np.mean((y - np.mean(y))**2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-6))

    noise_score = max(0, min(1, snr / 20))  # normalize

    return energy, pitch_var, noise_score


# ================= TEXT + TIMING =================
def analyze_text(file_path):
    segments, _ = model.transcribe(file_path)

    text = ""
    timestamps = []

    for seg in segments:
        text += seg.text + " "
        timestamps.append((seg.start, seg.end))

    text = text.strip()

    doc = nlp(text)
    words = [t.text for t in doc if t.is_alpha]
    total_words = len(words)

    duration = timestamps[-1][1] if timestamps else 1
    wpm = total_words / (duration / 60)

    # fillers
    lower = text.lower()
    filler_count = sum(lower.count(f) for f in FILLERS)

    # pauses
    pauses = 0
    for i in range(1, len(timestamps)):
        if timestamps[i][0] - timestamps[i-1][1] > 0.8:
            pauses += 1

    return text, total_words, wpm, filler_count, pauses


# ================= SCORING =================
def compute_scores(file_path):
    # AUDIO
    energy, pitch_var, noise_score = extract_audio_features(file_path)

    # TEXT
    text, words, wpm, fillers, pauses = analyze_text(file_path)

    # ================= CLARITY =================
    clarity = (
        0.4 * (1 - fillers / max(1, words)) +
        0.3 * min(1, energy * 10) +
        0.3 * max(0, 1 - pauses / 10)
    )

    # ================= SPEED =================
    speed = max(0, 1 - abs(wpm - IDEAL_WPM) / IDEAL_WPM)

    # ================= STRUCTURE =================
    doc = nlp(text)
    sentences = list(doc.sents)

    avg_len = np.mean([len(s) for s in sentences]) if sentences else 0
    structure = min(1, avg_len / 15)

    # ================= COMMUNICATION =================
    communication = (
        0.35 * clarity +
        0.25 * speed +
        0.20 * structure +
        0.20 * (pitch_var / (pitch_var + 50))
    )

    return {
        "clarity": float(round(clarity, 3)),
        "speed": float(round(speed, 3)),
        "structure": float(round(structure, 3)),
        "communication": float(round(communication, 3)),
        "noise": float(round(noise_score, 3))
    }


# ================= API =================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, file_id + ".wav")

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        scores = compute_scores(path)
    finally:
        os.remove(path)

    return scores


@app.get("/")
def root():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run("voice_analyzer:app", host="0.0.0.0", port=7500, reload=False)