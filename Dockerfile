# Multi-stage production build for ScamSentry (STAR-Fraud Engine)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and PyG dependencies first for caching efficiency
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric

# Install remaining project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository source code and artifacts
COPY . .

# Ensure artifacts and logs directories exist with write permissions
RUN mkdir -p artifacts/logs artifacts/saved_models data/processed

EXPOSE 8000 8501

# Default health check for FastAPI gateway
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/switch/health || exit 1

# Default entrypoint runs FastAPI gateway
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
