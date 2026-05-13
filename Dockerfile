# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# ffmpeg and libsndfile1 are required for librosa and audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download the spaCy English model
RUN python -m spacy download en_core_web_sm

# Pre-download the Whisper base model to bake it into the image (so it doesn't download on first run)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', compute_type='int8')"

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 7500

# Command to run the application
CMD ["uvicorn", "voice_analyzer:app", "--host", "0.0.0.0", "--port", "7500"]
