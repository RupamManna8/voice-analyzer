# Interview Audio Model / Voice Analyzer

A lightweight voice analysis service that extracts features, transcribes audio, scores responses, and returns insight metrics for interview-style audio recordings.

## Project layout
- `app/` — FastAPI application entry and API routes ([app/main.py](app/main.py)).
- `app/api/routes/analyze.py` — primary analysis endpoint ([app/api/routes/analyze.py](app/api/routes/analyze.py)).
- `services/` — domain services (transcription, scoring, sentiment, features).
- `schemas/` — request/response Pydantic schemas.

## Features
- Upload audio files and receive transcription, sentiment, scoring, and insights.
- Modular services so you can swap models or feature extractors.
- Docker-ready for easy deployment.

## Requirements
- Python 3.10+ (recommended)
- See `requirements.txt` for full dependency list.

## Quickstart (local)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app (development):

```powershell
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Automatic API docs (Swagger UI) live at `http://localhost:8000/docs`.

## API

POST `/api/analyze`
- Description: Analyze an uploaded audio file and return transcription, sentiment, scores, and insights.
- Endpoint implementation: [app/api/routes/analyze.py](app/api/routes/analyze.py)
- Request: `multipart/form-data` with `file` (audio file, e.g., WAV/MP3) and optional JSON fields specified in [schemas/request_schema.py](schemas/request_schema.py).
- Response: JSON described by [schemas/response_schema.py](schemas/response_schema.py).

### HTTP endpoints exposed by the service

- `POST /api/v1/analyze` - unified speech intelligence dashboard.
- `POST /api/v1/communication/analyze` - communication-focused analysis.
- `POST /api/v1/emotion/timeline` - emotion timeline analysis.
- `WS /ws/emotion-stream` - live emotion stream for PCM audio chunks.

The React frontend expects the service to be reachable at `http://localhost:7500` by default. Override that with `VITE_AUDIO_MODEL_URL` in the frontend environment if needed.

### Example curl

```bash
curl -X POST "http://localhost:7500/api/v1/analyze" \
	-F "file=@/path/to/answer.wav" \
	-H "accept: application/json"
```

## Docker

- Build image locally:

```powershell
docker build -t voice-analyzer:local .
```

- Run with docker-compose:

```powershell
docker-compose up --build
```

## Development notes
- Core configuration is in [core/config.py](core/config.py).
- Services live under `services/` (e.g., `services/transcription_service.py`, `services/scoring_service.py`).
- Add or replace ML models in the services directory; keep interfaces consistent.

## Contributing
- Open issues or PRs. Follow typical Python project conventions.

## License
- Add your license of choice here.

## Questions or changes
- Tell me if you want a longer README, badges, or CI instructions.

