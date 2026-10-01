# =============================================================================
# TelcoSense Platform - Production Multi-Stage Dockerfile
# =============================================================================
FROM python:3.11-slim as base

# Prevents Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    USE_TF=0 \
    USE_TORCH=1 \
    TRANSFORMERS_NO_TF=1

WORKDIR /app

# Install system build dependencies & curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, configuration, and data
COPY config/ /app/config/
COPY src/ /app/src/
COPY data/ /app/data/

# Expose FastAPI application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI Gateway via Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
