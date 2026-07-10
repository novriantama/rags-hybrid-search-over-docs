# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

# Copy dependency configs
COPY frontend/package*.json ./

# Install packages
RUN npm install

# Copy source code and compile production assets
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system utilities needed for compiling c-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

# Upgrade pip and install package dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Stage 3: Final production image
FROM python:3.12-slim

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy site-packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy codebase elements
COPY src/ ./src
COPY data/ ./data
COPY tests/ ./tests

# Copy pre-compiled React build folder
COPY --from=frontend-builder /frontend/dist/ ./frontend/dist/

# Set env variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
EXPOSE 8501

# Start the uvicorn API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
