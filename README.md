# Speech Intelligence API

Production-ready FastAPI backend for deterministic speech analysis on uploaded audio files. The service returns transcription, speech metrics, voice metrics, sentiment, communication scoring, and rule-based behavioral insights without using any LLM summarization or generative AI.

## Features

- `POST /api/v1/analyze` for multipart audio uploads
- `GET /` health endpoint
- Async FastAPI route layer with clean service boundaries
- Faster-Whisper transcription
- Librosa-based voice metrics
- Transformer-backed sentiment analysis with safe deterministic fallback
- Rule-based communication scoring and behavioral insights
- Upload size and file type validation
- Centralized exception handling
- CORS support and request timing headers
- Dockerized runtime

## Supported Formats

- `wav`
- `mp3`
- `m4a`

## Environment Variables

The application reads configuration from `.env` or the runtime environment.

- `PORT` - server port, default `7500`
- `MAX_UPLOAD_SIZE_MB` - maximum file size, default `25`
- `WHISPER_MODEL_SIZE` - Whisper model size, default `base`
- `WHISPER_COMPUTE_TYPE` - Whisper compute type, default `int8`
- `SENTIMENT_MODEL_NAME` - Hugging Face sentiment model name
- `CORS_ALLOW_ORIGINS` - comma-separated list, default `*`

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7500 --reload
```

## Docker

```bash
docker compose up --build
```

## API

### Health Check

```http
GET /
```

Response:

```json
{
  "status": "healthy",
  "service": "Speech Intelligence API",
  "version": "1.0.0"
}
```

### Analyze Audio

```http
POST /api/v1/analyze
Content-Type: multipart/form-data
```

Example `curl`:

```bash
curl -X POST "http://localhost:7500/api/v1/analyze" ^
  -F "file=@sample.wav" ^
  -F "language=en"
```

Example response:

```json
{
  "request_id": "3df51f6d-6f3d-4ac2-9f84-421be33c81df",
  "status": "success",
  "processing_time_ms": 1800,
  "metadata": {
    "filename": "sample.wav",
    "duration_sec": 90.2,
    "sample_rate": 16000,
    "channels": 1,
    "language": "en"
  },
  "transcript": {
    "text": "Hello everyone...",
    "segments": [
      {
        "start": 0.0,
        "end": 3.2,
        "text": "Hello everyone"
      }
    ]
  },
  "speech_metrics": {
    "total_words": 214,
    "words_per_minute": 136,
    "filler_words": {
      "total": 7,
      "details": {
        "um": 3,
        "uh": 2
      }
    },
    "pause_analysis": {
      "total_pauses": 6,
      "long_pauses": 2,
      "average_pause_sec": 1.1
    }
  },
  "voice_metrics": {
    "energy": 0.082,
    "pitch_variation": 108.3,
    "volume_stability": 0.79,
    "noise_score": 0.11
  },
  "sentiment_analysis": {
    "overall_sentiment": "positive",
    "sentiment_score": 0.84
  },
  "communication_analysis": {
    "clarity_score": 0.81,
    "fluency_score": 0.77,
    "confidence_score": 0.74,
    "pace_score": 0.69,
    "communication_score": 0.79
  },
  "behavioral_insights": {
    "strengths": ["Clear speech delivery"],
    "issues_detected": ["Frequent filler words"],
    "recommendations": ["Reduce filler word usage"]
  }
}
```

## OpenAPI / Swagger

Swagger UI is available at `/docs` and the OpenAPI spec is available at `/openapi.json`.

## Scalability Notes

The codebase is organized so future additions can be introduced without rewriting the API layer:

- WebSocket streaming can be added under `app/api/routes/`
- Speaker diarization and emotion detection can be added as new services
- Kafka, PostgreSQL, and Redis integrations can be isolated behind adapters
- Authentication and rate limiting can be added as middleware or dependencies
