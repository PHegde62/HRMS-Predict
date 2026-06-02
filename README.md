# HRMS·Predict — Ensemble Multi-Repository In Silico HRMS Metabolite Predictor

> A production-grade, locally-hosted Drug Metabolism and Pharmacokinetics (DMPK)
> platform that aggregates five independent prediction engines into a single
> ensemble workflow, computing high-resolution mass spectrometry (HRMS) adduct
> masses, delta-mass shifts, and P450 soft-spot vulnerabilities for any input
> parent compound.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Java / BioTransformer Troubleshooting](#5-java--biotransformer-troubleshooting)
6. [Running the Application](#6-running-the-application)
7. [API Reference](#7-api-reference)
8. [Git Setup and Version Control](#8-git-setup-and-version-control)
9. [Project Structure](#9-project-structure)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Overview

HRMS·Predict integrates five orthogonal metabolite prediction paradigms behind a
single FastAPI backend and a Streamlit frontend styled as premium pharmaceutical
analysis software.

| Engine | Repository | Role |
|---|---|---|
| **SyGMa** | `3D-e-Chem/sygma` | Phase I & II rule-based SMARTS reaction tree |
| **BioTransformer** | `Wishartlab-openscience/biotransformer3` | Multi-module mammalian + gut-microbial rules via JAR |
| **Meta-Predictor** | `zhukeyun/Meta-Predictor` | Deep-learning transformer site-of-metabolism (SoM) weights |
| **MetaTrans** | `KavrakiLab/MetaTrans` | SMILES→SMILES seq2seq structural translation |
| **SMARTCyp** | `cdk/smartcyp` | DFT-derived activation-energy P450 soft-spot profiling |

**Core outputs for every predicted metabolite:**

- Neutral Monoisotopic Mass (4 d.p., Da)
- Exact Delta Mass Δ*m/z* vs. parent (sign-prefixed)
- Theoretical `[M+H]+` and `[M-H]-` adduct masses (+/− 1.007276 Da)
- Per-atom soft-spot vulnerability scores cross-referenced across rule-based and DL engines
- Ensemble confidence score and consensus verification flag

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit Frontend  (port 8501)                        │
│  app/frontend.py                                        │
│  Ketcher sketcher · KPI cards · SVG map · LC-MS table   │
└───────────────────────┬─────────────────────────────────┘
                        │  HTTP  (localhost)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                           │
│  app/main.py                                            │
│  POST /predict · POST /render-soft-spots · GET /health  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Metabolism Engine  app/engine/metabolism.py            │
│                                                         │
│  Pipeline A: SyGMaPipeline                              │
│  Pipeline B: BioTransformerPipeline  (subprocess/JAR)   │
│  Pipeline C: MetaboliteTransformerPredictor  (PyTorch)  │
│  Pipeline D: SMARTCypProfiler  (SMARTS/DFT rules)       │
│                                                         │
│  MassSpecTracker → dedup (InChIKey) → ensemble score    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Prerequisites

| Dependency | Minimum Version | Notes |
|---|---|---|
| [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) | Any recent | Provides `conda` solver |
| Git | 2.34+ | For cloning and version control |
| Java (OpenJDK) | **17** | Managed by conda; required for BioTransformer JAR |
| NVIDIA GPU + CUDA | Optional | Only needed if running DL pipelines on GPU |

> **macOS Apple Silicon note:** the `torch` wheel in the environment targets CPU
> by default. To use MPS acceleration, set `dl_device = "mps"` in the frontend
> sidebar after installation.

---

## 4. Installation

### 4.1 Clone the repository

```bash
# Clone via HTTPS
git clone https://github.com/<your-username>/hrms-predict.git

# Or via SSH (recommended if you have an SSH key configured)
git clone git@github.com:<your-username>/hrms-predict.git

# Enter the project root
cd hrms-predict
```

### 4.2 Build the conda environment

The `environment.yml` file pins Python 3.10, installs RDKit and OpenJDK 17
from `conda-forge`, then delegates the remaining packages (FastAPI, Streamlit,
PyTorch, Transformers, SyGMa) to pip.

```bash
# Create the environment (takes 5–15 minutes on first run)
conda env create -f environment.yml

# Activate it
conda activate hrms-predictor

# Verify the Python interpreter resolves inside the env
which python
# Expected output: .../envs/hrms-predictor/bin/python
```

> If `conda env create` fails on the SyGMa git URL, ensure your machine can
> reach `github.com` and that `git` is on your PATH before running conda.

### 4.3 Download BioTransformer

BioTransformer is distributed as a standalone JAR and is **not** fetched by
`environment.yml` due to its binary size. Download it manually:

```bash
# Create the expected data directory
mkdir -p data/biotransformer

# Option A — download with curl (replace <version> with the latest release tag)
curl -L \
  "https://github.com/wishartlab/biotransformer3/releases/latest/download/BioTransformer3.0.jar" \
  -o data/biotransformer/BioTransformer3.0.jar

# Option B — download with wget
wget -O data/biotransformer/BioTransformer3.0.jar \
  "https://github.com/wishartlab/biotransformer3/releases/latest/download/BioTransformer3.0.jar"

# Confirm the file exists and is non-empty
ls -lh data/biotransformer/BioTransformer3.0.jar
# Expected: -rw-r--r--  1 user  group   ~120M  BioTransformer3.0.jar
```

### 4.4 Download DL model checkpoints

Place the MetaTrans and Meta-Predictor HuggingFace checkpoints under the
`models/` directory. Each model should be a standard HuggingFace checkpoint
folder (containing `config.json`, `pytorch_model.bin` or `model.safetensors`,
and `tokenizer.json`).

```bash
mkdir -p models/metatrans
mkdir -p models/metapredictor

# Example using HuggingFace CLI (pip install huggingface_hub if needed)
huggingface-cli download KavrakiLab/MetaTrans   --local-dir models/metatrans
huggingface-cli download zhukeyun/Meta-Predictor --local-dir models/metapredictor
```

### 4.5 Configure environment variables (optional)

Create a `.env` file at the project root to override default paths:

```bash
cp .env.example .env
```

Edit `.env` with your actual paths:

```dotenv
# Absolute path to BioTransformer JAR (overrides data/biotransformer/ default)
BIOTRANSFORMER_JAR=/absolute/path/to/BioTransformer3.0.jar

# Override default model directories
METATRANS_MODEL_DIR=/absolute/path/to/models/metatrans
METAPREDICTOR_MODEL_DIR=/absolute/path/to/models/metapredictor
```

---

## 5. Java / BioTransformer Troubleshooting

BioTransformer requires a working Java 17 runtime on the system PATH **inside
the active conda environment**. Work through this checklist top-to-bottom until
every check prints the expected output.

### Check 1 — Confirm the conda env is active

```bash
conda info --envs
```

The active environment is marked with a `*`. You must see `hrms-predictor *`
before proceeding. If not:

```bash
conda activate hrms-predictor
```

### Check 2 — Confirm `java` is on PATH

```bash
which java
```

**Expected output (Linux/macOS):**

```
/home/<user>/miniconda3/envs/hrms-predictor/bin/java
```

**Expected output (Windows, Git Bash / PowerShell):**

```
C:\Users\<user>\miniconda3\envs\hrms-predictor\Scripts\java.exe
```

If `which java` returns nothing or points outside the conda env, the OpenJDK
install did not complete. Reinstall it:

```bash
conda install -c conda-forge openjdk=17 --force-reinstall
```

### Check 3 — Verify Java version

```bash
java -version
```

**Expected output:**

```
openjdk version "17.x.x" 20xx-xx-xx
OpenJDK Runtime Environment (build 17.x.x+x)
OpenJDK 64-Bit Server VM (build 17.x.x+x, mixed mode)
```

If you see version 8 or 11, an older Java installation on the system PATH is
shadowing the conda-managed one. Fix this by ensuring the conda env bin
directory appears **first** in `$PATH`:

```bash
# Inspect PATH order
echo $PATH | tr ':' '\n' | head -10

# If the conda env bin is not first, prepend it explicitly for this session:
export PATH="$(conda info --base)/envs/hrms-predictor/bin:$PATH"

# To make it permanent, add the export to your shell profile (~/.bashrc,
# ~/.zshrc, or ~/.profile) AFTER the conda init block.
```

### Check 4 — Verify `JAVA_HOME` points to the conda-managed JDK

BioTransformer's JAR launcher may read `JAVA_HOME` directly. Set it to the
conda env's JDK root:

```bash
# Locate the conda JDK
JAVA_HOME_CANDIDATE="$(dirname $(dirname $(which java)))"
echo $JAVA_HOME_CANDIDATE
# Should print: .../envs/hrms-predictor

# Set it for the current session
export JAVA_HOME="$JAVA_HOME_CANDIDATE"

# Verify
echo $JAVA_HOME
java -XshowSettings:all -version 2>&1 | grep "java.home"
```

To persist `JAVA_HOME` across sessions, add the export to your shell profile
directly after the `conda activate hrms-predictor` line or in a `conda`
activation hook:

```bash
# Create a conda activation hook (runs automatically on conda activate)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/java_home.sh" << 'EOF'
export JAVA_HOME="$(dirname $(dirname $(which java)))"
EOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/java_home.sh"
```

### Check 5 — Run a smoke test against the JAR

```bash
java -Xmx512m -jar data/biotransformer/BioTransformer3.0.jar -h
```

**Expected:** BioTransformer prints its usage/help text.

**If you see** `Error: Unable to access jarfile`:

```bash
# The JAR path is wrong or the file is missing. Recheck:
ls -lh data/biotransformer/BioTransformer3.0.jar

# Set the env var override explicitly
export BIOTRANSFORMER_JAR="$(pwd)/data/biotransformer/BioTransformer3.0.jar"
```

**If you see** `UnsupportedClassVersionError`:

The JAR was compiled for a newer Java than the one currently active. Upgrade
to Java 17+ as described in Check 3.

### Check 6 — Run the engine's built-in path resolution test

```bash
python - << 'EOF'
import sys
sys.path.insert(0, '.')
from app.engine.metabolism import BioTransformerPipeline

try:
    pipe = BioTransformerPipeline()
    print(f"JAR resolved to: {pipe.jar_path}")
    print("BioTransformerPipeline initialised successfully.")
except FileNotFoundError as e:
    print(f"FAIL — {e}")
EOF
```

If this prints the JAR path, BioTransformer is fully configured.

---

## 6. Running the Application

The backend and frontend are two separate processes. Open **two terminal
windows** (or two split panes in tmux / VS Code), activate the conda
environment in each, then run one command per window.

### Terminal Window 1 — FastAPI backend

```bash
# Navigate to the project root
cd hrms-predict

# Activate the environment
conda activate hrms-predictor

# Start the Uvicorn server on port 8000 with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected startup output:**

```
INFO:     Will watch for changes in these directories: ['/path/to/hrms-predict']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The interactive API documentation (Swagger UI) is available at:

```
http://localhost:8000/docs
```

The ReDoc alternative is at:

```
http://localhost:8000/redoc
```

### Terminal Window 2 — Streamlit frontend

```bash
# New terminal window — navigate to project root
cd hrms-predict

# Activate the environment (must be done in every new terminal)
conda activate hrms-predictor

# Launch Streamlit on port 8501
streamlit run app/frontend.py --server.port 8501 --server.address localhost
```

**Expected startup output:**

```
  You can now view your Streamlit app in your browser.

  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The browser tab opens automatically. If it does not, navigate to:

```
http://localhost:8501
```

### Running both servers with a single command (optional)

If you prefer launching from one terminal, you can use a background process:

```bash
# Start backend in background, capture its PID
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Start frontend in foreground
streamlit run app/frontend.py --server.port 8501 --server.address localhost

# When done, stop the backend
kill $BACKEND_PID
```

Or with **tmux** (split panes):

```bash
tmux new-session -d -s hrms -x 220 -y 50
tmux send-keys -t hrms "conda activate hrms-predictor && uvicorn app.main:app --reload --port 8000" Enter
tmux split-window -h -t hrms
tmux send-keys -t hrms "conda activate hrms-predictor && streamlit run app/frontend.py --server.port 8501" Enter
tmux attach -t hrms
```

### Stopping the servers

In each terminal window, press `Ctrl + C` to gracefully shut down the process.

---

## 7. API Reference

### `POST /predict`

Run the full ensemble prediction pipeline against a parent SMILES.

**Request body (JSON):**

```json
{
  "smiles":               "CN1C(=O)CN=C(c2ccccc2Cl)c2cc(Cl)cnn21",
  "run_sygma":            true,
  "run_biotransformer":   true,
  "run_dl":               true,
  "run_smartcyp":         true,
  "biotransformer_type":  "allHuman",
  "dl_device":            "cpu",
  "sygma_phase1_cycles":  1,
  "sygma_phase2_cycles":  1,
  "smartcyp_ea_cutoff":   95.0
}
```

**Response (abridged):**

```json
{
  "parent": {
    "smiles":           "CN1C(=O)CN=C(c2ccccc2Cl)c2cc(Cl)cnn21",
    "neutral_mass":     325.0305,
    "molecular_formula":"C15H11Cl2N5O",
    "adducts": { "mplus_h": 326.0378, "mminus_h": 324.0232 }
  },
  "metabolites": [
    {
      "rank":              1,
      "smiles_canonical":  "...",
      "neutral_mass":      341.0254,
      "delta_mass":        "+15.9949",
      "adducts":           { "mplus_h": 342.0327, "mminus_h": 340.0181 },
      "source_pipeline":   "sygma|biotransformer|dl",
      "confidence_label":  "High Confidence (Consensus Verified)",
      "ensemble_score":    0.8750
    }
  ],
  "pipeline_stats": {
    "sygma_count": 12, "biotransformer_count": 9,
    "dl_count": 5,     "smartcyp_count": 1,
    "total_after_dedup": 18, "consensus_count": 4,
    "elapsed_seconds": 47.3
  }
}
```

### `POST /render-soft-spots`

Return an SVG with per-atom glow overlays scaled by risk score.

**Request body (JSON):**

```json
{
  "smiles": "CN1C(=O)CN=C(c2ccccc2Cl)c2cc(Cl)cnn21",
  "atom_scores": [
    { "atom_idx": 3,  "score": 0.92, "isoform": "CYP3A4" },
    { "atom_idx": 11, "score": 0.65, "isoform": "CYP2D6" }
  ],
  "width":               560,
  "height":              380,
  "highlight_alpha_max": 0.70,
  "colour_scheme":       "risk"
}
```

**Response:** `image/svg+xml` string.

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0"}
```

---

## 8. Git Setup and Version Control

### 8.1 Initialise a local git repository

```bash
# From the project root
cd hrms-predict

# Initialise git (if not already a git repo)
git init

# Set your identity (if not already configured globally)
git config user.name  "Your Name"
git config user.email "you@example.com"
```

### 8.2 Create a `.gitignore`

Before staging any files, create a `.gitignore` to exclude environment artefacts,
large binary files, and secrets:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
.Python
*.egg-info/
dist/
build/
.eggs/

# Conda / virtual envs
envs/
.conda/
*.conda

# Environment secrets
.env
*.env.local

# Model weights (large binaries — track with Git LFS if needed)
models/

# BioTransformer JAR (large binary)
data/biotransformer/*.jar
data/biotransformer/*.zip

# Jupyter notebooks checkpoints
.ipynb_checkpoints/

# macOS
.DS_Store

# IDE
.vscode/
.idea/
*.swp
*.swo

# Test artefacts
.pytest_cache/
htmlcov/
.coverage

# Streamlit
.streamlit/secrets.toml
EOF
```

### 8.3 Stage and commit all source files

```bash
# Review exactly what will be staged (dry run)
git status

# Stage everything (gitignore filters out the large files above)
git add .

# Verify the staged file list looks correct before committing
git status

# Create the initial commit
git commit -m "feat: initial HRMS Predictor stack

- app/engine/metabolism.py  — five-pipeline metabolism engine
- app/main.py               — FastAPI backend with /predict, /render-soft-spots
- app/frontend.py           — Streamlit dark-slate pharmaceutical UI
- environment.yml           — reproducible conda environment (Python 3.10)
- README.md                 — full setup and operational documentation"
```

### 8.4 Create a new repository on GitHub

1. Open [github.com/new](https://github.com/new) in your browser.
2. Set **Repository name** to `hrms-predict`.
3. Leave it **Private** unless you intend to publish (model weights and JAR
   paths may contain sensitive data).
4. Do **not** tick *Add a README*, *Add .gitignore*, or *Choose a license* —
   you already have all of these locally.
5. Click **Create repository**.

### 8.5 Connect the local repo to GitHub and push

GitHub will display the exact commands after you create the repo. They follow
this pattern:

```bash
# Add the remote (replace <your-username> with your GitHub handle)
git remote add origin git@github.com:<your-username>/hrms-predict.git

# Rename the default branch to 'main' if git init created 'master'
git branch -M main

# Push and set the upstream tracking branch
git push -u origin main
```

Confirm the push succeeded:

```bash
git log --oneline origin/main
# Should print your initial commit hash and message
```

### 8.6 Handling large files (model weights, JAR)

If you want to track model weights or the BioTransformer JAR in the repository,
use [Git Large File Storage (LFS)](https://git-lfs.com/) rather than committing
them directly. Large binaries committed to git permanently inflate the
repository and cannot be cleanly removed from history.

```bash
# Install Git LFS (once per machine)
git lfs install

# Track .jar and model weight file extensions
git lfs track "*.jar"
git lfs track "*.bin"
git lfs track "*.safetensors"
git lfs track "*.pt"
git lfs track "*.pth"

# Stage the .gitattributes file that Git LFS created
git add .gitattributes
git commit -m "chore: configure Git LFS for large binaries"

# Now add and commit the large files normally — LFS handles the rest
git add data/biotransformer/BioTransformer3.0.jar
git add models/
git commit -m "feat: add BioTransformer JAR and DL model checkpoints"
git push
```

> **GitHub LFS bandwidth:** the free tier includes 1 GB of storage and 1 GB/month
> of bandwidth. For production use, consider hosting weights on HuggingFace Hub
> and symlinking or downloading on first run.

### 8.7 Recommended branch workflow

```bash
# Create a feature branch for any new engine or UI work
git checkout -b feat/add-reaxys-pipeline

# Work, commit incrementally...
git add app/engine/reaxys_wrapper.py
git commit -m "feat(engine): add Reaxys reaction prediction wrapper"

# Push the feature branch
git push -u origin feat/add-reaxys-pipeline

# Open a Pull Request on GitHub, review, then merge into main
# After merging, clean up locally:
git checkout main
git pull origin main
git branch -d feat/add-reaxys-pipeline
```

---

## 9. Project Structure

```
hrms-predict/
├── environment.yml                  # Reproducible conda environment
├── README.md                        # This document
├── .env.example                     # Environment variable template
├── .gitignore                       # Excludes envs, secrets, large binaries
│
├── app/
│   ├── main.py                      # FastAPI server — endpoints + SVG renderer
│   ├── frontend.py                  # Streamlit UI — dark-slate SaaS interface
│   └── engine/
│       └── metabolism.py            # Core 5-pipeline metabolism engine
│
├── models/
│   ├── metatrans/                   # HuggingFace seq2seq checkpoint (not tracked)
│   └── metapredictor/               # HuggingFace token-classification checkpoint
│
└── data/
    ├── biotransformer/
    │   └── BioTransformer3.0.jar    # Downloaded separately (not tracked by git)
    └── smartcyp/
        └── SMARTCyp_SCORE_DATA.tsv  # DFT fragment activation energies
```

---

## 10. Acknowledgements

This tool builds on the published work of the following research groups and
open-source projects:

- **SyGMa** — Ridder, L. & Wagener, M. (2008). *SyGMa: Combining Expert Knowledge
  and Empirical Scoring in the Prediction of Metabolites.* ChemMedChem.
  [`3D-e-Chem/sygma`](https://github.com/3D-e-Chem/sygma)

- **BioTransformer** — Djoumbou-Feunang, Y. et al. (2019). *BioTransformer: A
  comprehensive computational tool for small molecule metabolism prediction and
  metabolite identification.* J. Cheminformatics.
  [`Wishartlab-openscience/biotransformer3`](https://github.com/Wishartlab-openscience/biotransformer3)

- **MetaTrans** — Litsa, E.E. et al. (2023). *An end-to-end deep learning framework
  for computational prediction of cytochrome P450 2D6-mediated drug metabolism.*
  [`KavrakiLab/MetaTrans`](https://github.com/KavrakiLab/MetaTrans)

- **Meta-Predictor** — Zhu, K. et al. *Meta-Predictor: A transformer-based site of
  metabolism predictor.*
  [`zhukeyun/Meta-Predictor`](https://github.com/zhukeyun/Meta-Predictor)

- **SMARTCyp** — Rydberg, P. et al. (2010). *SMARTCyp: A 2D Method for Prediction
  of Cytochrome P450-Mediated Drug Metabolism.* J. Chem. Inf. Model.

- **RDKit** — An open-source cheminformatics toolkit. [`rdkit.org`](https://www.rdkit.org)

- **FastAPI** — [`fastapi.tiangolo.com`](https://fastapi.tiangolo.com)

- **Streamlit** — [`streamlit.io`](https://streamlit.io)

---

*HRMS·Predict is intended for research use only and does not constitute
regulatory or clinical advice. Metabolite predictions are computational estimates
and should be validated experimentally.*
