.PHONY: help install test ingest seed chunk dense-search sparse-search rrf-search rerank-search run run-api run-frontend run-dashboard docker-build docker-up docker-down docker-logs clean

# Default shell
SHELL := /bin/bash

# Python environment variables
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest

help:
	@echo "=========================================================================="
	@echo "                    Hybrid RAG Project Makefile Helper                    "
	@echo "=========================================================================="
	@echo "Available commands:"
	@echo "  make install         - Initialize virtual environment & install requirements"
	@echo "  make test            - Execute all unit tests"
	@echo "  make clean           - Clear temporary cache files & logs"
	@echo ""
	@echo "Pipeline Execution Runs:"
	@echo "  make ingest          - Parse and ingest raw documents to data/processed"
	@echo "  make seed            - Clear and seed RAG database with documentation corpus"
	@echo "  make chunk           - Run chunker comparison (Fixed, Recursive, Semantic)"
	@echo "  make dense-search    - Run dense retrieval search against ChromaDB"
	@echo "  make sparse-search   - Run sparse retrieval search against BM25"
	@echo "  make rrf-search      - Run RRF hybrid retrieval search scenarios"
	@echo "  make rerank-search   - Run second-pass Cross-Encoder reranking"
	@echo ""
	@echo "Running the Application:"
	@echo "  make run             - Run API and React dashboard concurrently"
	@echo "  make run-api         - Start FastAPI backend server"
	@echo "  make run-frontend    - Start Vite React frontend server"
	@echo "  make run-dashboard   - Start Streamlit bridge page"
	@echo ""
	@echo "Docker Operations:"
	@echo "  make docker-build    - Build app image using Dockerfile"
	@echo "  make docker-up       - Run ChromaDB, API, and Dashboard containers"
	@echo "  make docker-down     - Stop and clean Docker Compose containers"
	@echo "  make docker-logs     - Tail docker container logs"
	@echo "=========================================================================="

$(PYTHON):
	@echo "Creating virtual environment in $(VENV_DIR)..."
	python3 -m venv $(VENV_DIR)

install: $(PYTHON)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Installing project dependencies..."
	$(PIP) install -e .
	@echo "Setup complete. Virtual environment ready."

test:
	@echo "Executing pytest unit test suite..."
	$(PYTEST)

ingest:
	@echo "Running document ingestion loading pipeline..."
	$(PYTHON) scratch/test_ingestion_runner.py

seed:
	@echo "Seeding RAG database with documentation corpus..."
	$(PYTHON) -m src.ingestion.seed

chunk:
	@echo "Running chunking comparison report..."
	$(PYTHON) -m src.evaluation.compare_chunking

dense-search:
	@echo "Running vector dense retrieval lookup..."
	$(PYTHON) scratch/test_dense_retrieval.py

sparse-search:
	@echo "Running BM25 sparse keyword lookup..."
	$(PYTHON) scratch/test_sparse_retrieval.py

rrf-search:
	@echo "Running Reciprocal Rank Fusion scenario comparisons..."
	$(PYTHON) scratch/test_rrf_fusion.py

rerank-search:
	@echo "Running neural Cross-Encoder re-ranking refinement..."
	$(PYTHON) scratch/test_reranking.py

docker-build:
	@echo "Building docker containers..."
	docker compose build

docker-up:
	@echo "Starting container services in the background (ChromaDB, API Server, Dashboard)..."
	docker compose up -d
	@echo "Containers started. Run 'make docker-logs' to monitor logs."

docker-down:
	@echo "Stopping container services..."
	docker compose down

docker-logs:
	@echo "Tailing container logs..."
	docker compose logs -f

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Cleaning localized temporary indices..."
	rm -rf data/chroma/*
	rm -f data/bm25/bm25_index.pkl
	@echo "Clean completed successfully."

run:
	@echo "Starting API backend and Frontend dashboard..."
	@echo "Open http://localhost:5173 to view the dashboard."
	@echo "Press Ctrl+C to stop both."
	trap 'kill 0' EXIT; $(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & npm --prefix frontend run dev

run-api:
	@echo "Starting FastAPI backend server..."
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	@echo "Starting Vite React frontend server..."
	npm --prefix frontend run dev

run-dashboard:
	@echo "Starting Streamlit dashboard bridge page..."
	$(PYTHON) -m streamlit run src/dashboard/app.py --server.port 8501
