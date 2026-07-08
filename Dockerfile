# Build Stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies required for building C extensions (e.g. for certain numpy or vector libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package config files
COPY pyproject.toml README.md ./

# Install python dependencies first to cache this layer
# Note: we use --no-cache-dir to minimize image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Final Stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY src/ ./src
COPY data/ ./data
COPY tests/ ./tests

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command starts the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
