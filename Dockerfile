# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ULTRALYTICS_NO_FASTSAM=1
ENV PORT=8080

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads uploads/dataset

# Expose port (use PORT env var or default to 8080)
EXPOSE ${PORT:-8080}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:${PORT:-8080}/healthz')" || exit 1

# Run with Gunicorn (optimized for memory)
CMD python -m gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers=1 \
    --threads=1 \
    --timeout=120 \
    --max-requests=1000 \
    --max-requests-jitter=100 \
    --graceful-timeout=30
