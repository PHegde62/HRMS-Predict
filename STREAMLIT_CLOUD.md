# Deploying HRMS-Predict to Streamlit Community Cloud (free)

This hosts the **full tool** for free with no install and no billing. The app
runs the complete engine (SyGMa dual-track, SMARTCyp soft-spots, the
improvements, prioritization, CDD lookup, PDF report) **in-process** inside the
single Streamlit process — no second server, no ports.

## How this was made to work
Streamlit Community Cloud only runs one Streamlit process and installs deps from
a plain `requirements.txt` (no custom pip flags). Two things were done so the
full engine runs there:

1. **SyGMa is vendored** into the repo at `./sygma/` (pure Python + rule files).
   SyGMa has no wheel and its normal install needs RDKit + `--no-build-isolation`,
   which Streamlit Cloud can't do — vendoring sidesteps the build entirely.
2. **The frontend runs the FastAPI engine in-process** via a Starlette
   TestClient when no HTTP backend is reachable (auto-detected). Locally, if you
   still run `uvicorn`, it uses that instead. Force in-process with env var
   `HRMS_INPROCESS=1`.

`torch`/`transformers` are intentionally left out of `requirements.txt` to stay
within Streamlit Cloud's memory limit; the DL pipeline is inactive without model
weights anyway. The torch import is guarded, so the engine runs fine without it.

## Repo requirement
Streamlit Cloud deploys from a GitHub repo where the app sits at/near the root
(it reads `requirements.txt` + `packages.txt` from the repo root). This project's
root already has both files plus `app/frontend.py`, so deploy a repo whose root
**is** this project.

- **Easiest:** use `PHegde62/HRMS-Predict` (your existing repo). It's currently
  archived — unarchive it (GitHub → repo → Settings → scroll down → "Unarchive"),
  then push:
  ```powershell
  cd C:\Users\pooja_genesistherape\hrms-predict
  git add -A
  git commit -m "Cloud hosting: vendor SyGMa, in-process engine, Streamlit Cloud config"
  git push origin main
  ```
- The shared logbook (`genesistherapeutics/logbooks`) is a monorepo with the app
  nested deep, so it's a poor fit for Streamlit Cloud's root-level dependency
  detection — use a dedicated repo instead.

## Deploy steps
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Authorize Streamlit to read the repo (grant access to private repos if
   `HRMS-Predict` is private).
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `PHegde62/HRMS-Predict`
   - **Branch:** `main`
   - **Main file path:** `app/frontend.py`
5. (Optional) Advanced settings → Python version 3.10 or 3.11.
6. Click **Deploy**. First build takes a few minutes (RDKit wheel + deps).
7. You get a public URL like `https://hrms-predict-xxxx.streamlit.app` — that's
   your shareable link.

## First run
The engine imports RDKit on first prediction, so the very first click may take
10-20 s. After that it's warm. If you see "Backend offline" it just means the
in-process client hasn't initialized yet — click Analyse and it will.

## Access control
Streamlit Community Cloud apps are public by default, but you can restrict access
to specific Google/GitHub emails: app → Settings → Sharing → "Who can view this
app" → add allowed emails. Recommended if the tool must stay internal to Genesis.

## If the build ever fails
- **On `sygma`:** it should not appear in `requirements.txt` (it's vendored). If
  you re-added it, remove it.
- **Memory / resource limits:** don't add `torch`/`transformers` to
  `requirements.txt`; they blow the memory budget.
- **Fallback:** the `Dockerfile` in this repo runs the identical app on any
  Docker host (Hugging Face Spaces free Docker Space, Render, Railway, Fly.io)
  if you ever outgrow Streamlit Cloud's limits.

## Files that make this work
- `sygma/` — vendored SyGMa package (do not delete)
- `requirements.txt` — lean deps (no sygma/torch/transformers)
- `packages.txt` — apt deps (Java for BioTransformer, RDKit draw libs)
- `app/frontend.py` — main entry; auto-runs the engine in-process
- `app/main.py`, `app/engine/` — the engine, reused verbatim in-process
