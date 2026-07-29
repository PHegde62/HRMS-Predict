#!/usr/bin/env bash
# HRMS-Predict container entrypoint.
# Cloud Run injects $PORT (default 8080). Streamlit is the PUBLIC process on
# $PORT; the FastAPI engine runs internally on :8000 and the frontend reaches
# it via HRMS_BACKEND_URL=http://localhost:8000.
set -euo pipefail

PORT="${PORT:-8080}"
export HRMS_BACKEND_URL="http://localhost:8000"

# 1) Start the FastAPI backend (full engine) in the background on :8000.
#    RDKit/torch import can take ~15-30s on first boot; the frontend polls
#    /health and shows "backend offline" until it is ready.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 &

# 2) Start Streamlit on the public $PORT as the container's main process so the
#    port opens quickly and Cloud Run's startup probe passes.
exec streamlit run app/frontend.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
