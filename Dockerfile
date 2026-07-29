# HRMS-Predict — single-container image for Google Cloud Run (or any Docker host:
# Hugging Face Spaces, Render, Railway, Fly.io).
# Runs the full engine (FastAPI on :8000) + the Streamlit UI (public $PORT).
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HRMS_BACKEND_URL=http://localhost:8000 \
    PORT=8080

# System deps:
#   openjdk-17  -> BioTransformer JAR runtime (Java)
#   libxrender/libxext -> RDKit drawing
#   curl        -> health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        libxrender1 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install --upgrade pip \
    # CPU-only PyTorch + transformers enable the DL pipeline (if weights are
    # added under models/). Installed separately to keep requirements.txt lean.
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install transformers>=4.41.0 \
    && pip install -r requirements.txt

# --- Application code ---
# SyGMa is vendored at ./sygma (no pip build needed).
COPY sygma/ ./sygma/
COPY app/ ./app/
COPY models/ ./models/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8080
CMD ["./entrypoint.sh"]
