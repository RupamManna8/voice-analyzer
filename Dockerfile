# Use an official Python runtime as a parent image
FROM python:3.11.6-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    XDG_CACHE_HOME=/models/cache \
    HF_HOME=/models/hf_cache

WORKDIR /app

# Install system dependencies needed for audio processing and basic tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libsndfile1 \
    build-essential \
    python3-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies and install (keep image layers cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user and model/cache directories with correct ownership
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /models /app/app/temp && \
    chown -R app:app /models /app

# Copy application source
COPY . .

# Switch to non-root user for runtime
USER app

# Expose service port
EXPOSE 7500

# Healthcheck uses the app root which returns a JSON health payload
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://127.0.0.1:7500/ || exit 1

# Runtime environment defaults
ENV HOST=0.0.0.0 PORT=7500 WORKERS=4

# Start the app with configurable workers
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
