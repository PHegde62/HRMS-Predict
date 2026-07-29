# Deploying HRMS-Predict to Google Cloud Run

This hosts the **full tool** (FastAPI engine + Streamlit UI) as a single
container on Cloud Run, giving you a public HTTPS link. Cloud Run scales to zero
when idle, so you only pay for build storage + actual usage.

> You run these commands on your own machine (or Cloud Shell). I can't run
> `gcloud` against your Google Cloud account. Skip any step you've already done.

---

## What gets deployed
- One container: Streamlit UI on the public port, FastAPI engine on internal `:8000`.
- Pipelines included: SyGMa (Phase I/II, dual-track), SMARTCyp soft-spots, the
  N-dealkylation/reduction/glucuronidation improvements, prioritization, CDD
  lookup, PDF report.
- BioTransformer: Java runtime is in the image; drop the JAR in (see bottom) to enable it.
- DL (MetaTrans/Meta-Predictor): code + torch are included, but `models/` has no
  weights, so it stays inactive until you add them.

---

## 0. Prerequisites (one time)
1. A Google account with access to Google Cloud.
2. Install the gcloud CLI: https://cloud.google.com/sdk/docs/install
   Then authenticate:
   ```bash
   gcloud auth login
   ```
   (No local install? Use Cloud Shell at https://shell.cloud.google.com — gcloud
   is preinstalled; upload this repo folder there.)

## 1. Create (or pick) a project
```bash
gcloud projects create hrms-predict-app --name="HRMS-Predict"
gcloud config set project hrms-predict-app
```
(If you already have a project, just run the `config set project` line with its ID.)

## 2. Enable billing
Cloud Run + Cloud Build require billing enabled on the project. Do it once in the
console: https://console.cloud.google.com/billing → link the project to a billing
account. (Free-tier credits usually cover a tool like this.)

## 3. Enable the required APIs
```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com
```

## 4. Give the build enough time (torch makes the image large)
```bash
gcloud config set builds/timeout 3000s
```

## 5. Deploy (build + push + run, all from the Dockerfile)
From the repo root (`C:\Users\pooja_genesistherape\hrms-predict`):
```bash
gcloud run deploy hrms-predict \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --concurrency 8 \
  --min-instances 0 \
  --session-affinity
```
- `--source .` uses the `Dockerfile` here; Cloud Build builds and pushes it for you.
- `--allow-unauthenticated` makes the link publicly reachable (remove it to require Google sign-in).
- `--session-affinity` keeps Streamlit's websocket on one instance.
- First build takes ~10-15 min (torch + rdkit). Later deploys are faster (cached layers).

When it finishes, gcloud prints:
```
Service URL: https://hrms-predict-xxxxxxxx-uc.a.run.app
```
**That URL is your shareable link.**

## 6. First load
The backend needs ~15-30s to import RDKit/torch on a cold start. If the UI shows
"Backend offline" briefly, wait a few seconds and click Analyse again. To avoid
cold starts entirely, keep one instance warm (small always-on cost):
```bash
gcloud run services update hrms-predict --region us-central1 --min-instances 1
```

## 7. Redeploy after code changes
Re-run the **step 5** command. Same service, new revision, zero downtime.

---

## Access control options
- **Public link (default above):** anyone with the URL can use it.
- **Restrict to your org / specific people:** drop `--allow-unauthenticated`, then
  grant `roles/run.invoker` to specific users, or put it behind Identity-Aware
  Proxy (IAP). Recommended if the tool must stay internal to Genesis.

## Enabling BioTransformer (optional)
1. Put the BioTransformer JAR (+ its `database.zip`) under e.g. `app/engine/biotransformer/`.
2. Make sure `.gcloudignore` / `.dockerignore` don't exclude it (they exclude
   `*.pdf/pptx/docx` and `*.local.json`, not JARs).
3. Point the engine's BioTransformer path at it and redeploy. Java 17 is already installed.

## Cost & security notes
- Scales to zero: no compute cost when idle; you pay for the built image storage
  (a few cents/month) and per-request compute.
- The `.gcloudignore` keeps confidential GEN structures, internal PDFs/PPTX/DOCX,
  and the docs folder OUT of the upload. Keep it that way.
- The CDD Vault token is entered in the UI at runtime and never stored in the
  image — good. Don't bake tokens into env vars.
