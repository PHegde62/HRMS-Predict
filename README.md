<div align="center">

# HRMS·Predict

**In silico Phase I & II metabolite prediction with HRMS target list generation**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20518280.svg)](https://doi.org/10.5281/zenodo.20518280)
[![RRID](https://img.shields.io/badge/RRID-SCR__028518-orange)](https://scicrunch.org/resolver/SCR_028518)
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)](https://python.org)

*Developed by [Genesis Molecular AI](https://genesismolecularai.com)*

</div>

---

## Overview

HRMS·Predict is a free, open-source, locally-installable platform for predicting the Phase I and Phase II metabolites of drug-like compounds. It combines five prediction engines into a single ensemble pipeline, generates HRMS-grade exact masses for every predicted metabolite, and produces an enzyme-annotated soft-spot atom map — all running on your local hardware with no data leaving your machine.

**Key capabilities:**

- **Ensemble prediction** — SyGMa (Phase I/II rules), SMARTCyp (69 SMARTS rules across 11 enzyme classes), BioTransformer, MetaTrans, and Meta-Predictor combined with InChIKey deduplication and consensus scoring
- **HRMS mass annotation** — [M+H]⁺ and [M−H]⁻ adduct masses to 4 decimal places (0.0000 ppm theoretical error)
- **Soft-spot visualisation** — enzyme-coloured SVG atom map showing predicted metabolic sites
- **One-click export** — `.xlsx` LC-HRMS target list, `.csv`, and multi-page **PDF report** with structures
- **Batch prediction** — REST API endpoint for up to 100 compounds per request
- **Fully local** — FastAPI backend + Streamlit GUI, no internet required after install

---

## Benchmark performance

| Metric | Value | Dataset |
|---|---|---|
| Pooled recall | **63.0%** | MetXBioDB (1,245 substrates, 2,102 metabolites) |
| Recall @ top-10 | **53.1%** | MetXBioDB |
| Precision @ top-10 | **21.4%** | MetXBioDB |
| F1 @ top-10 | **0.305** | MetXBioDB |
| 10-compound panel recall | **27/29 (93%)** | CYP1A2, CYP2C9, CYP2D6, CYP2C19, CYP3A4, UGT, SULT, NAT2, AO |
| HRMS mass accuracy | **10/10, 0.0000 ppm** | 10 reference compounds, 129–558 Da |
| Mean prediction time | **~2.2 s/compound** | Intel Core i7, 16 GB RAM |

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/PHegde62/HRMS-Predict.git
cd HRMS-Predict
conda env create -f environment.yml
conda activate hrms-predictor
```

Start the tool:
```bash
# Terminal 1 — backend
uvicorn app.main:app --port 8000

# Terminal 2 — frontend
streamlit run app/frontend.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

### Option B — Docker

```bash
git clone https://github.com/PHegde62/HRMS-Predict.git
cd HRMS-Predict
docker-compose up
```

- Streamlit UI → `http://localhost:8501`
- FastAPI docs → `http://localhost:8000/docs`

### Requirements

- Python 3.10
- RDKit ≥ 2022.09
- 4 GB RAM minimum (8 GB recommended for large molecules)
- Windows 10/11, macOS, or Linux

---

## Quick start

### GUI (Streamlit)

1. Open `http://localhost:8501`
2. Draw or paste a SMILES string (e.g. `O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl` for diclofenac)
3. Click **Predict**
4. View the soft-spot map, ranked metabolite table, and HRMS masses
5. Click **Export .xlsx** for the LC-HRMS target list
6. Click **Generate PDF Report** for a full multi-page report with structures

### REST API

Single compound:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl"}'
```

Batch (up to 100 compounds):
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"compounds": [
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"},
    {"smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "name": "Caffeine"}
  ]}'
```

Batch → CSV target list:
```bash
curl -X POST http://localhost:8000/predict/batch/csv \
  -H "Content-Type: application/json" \
  -d '{"compounds": [{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"}]}' \
  -o target_list.csv
```

Full API documentation: `http://localhost:8000/docs`

---

## PDF report

Generate a standalone PDF report from any prediction output:

```bash
python generate_report.py your_output.xlsx --compound "Diclofenac" --top 12
```

The report includes:
- Cover page with parent structure and physicochemical properties
- Soft-spot analysis with enzyme-coloured atom map and legend
- Individual metabolite cards with structure, enzyme badge, HRMS masses, and likelihood
- Summary reference table ready for LC-HRMS import

---

## Enzyme colour coding

| Enzyme class | Colour |
|---|---|
| CYP (all isoforms) | Amber |
| UGT | Teal |
| SULT | Blue |
| NAT / COMT | Purple |
| FMO / AO | Coral |
| MAO | Dark red |
| GST | Green |
| Spontaneous | Grey |

---

## Supported enzyme classes

CYP3A4 · CYP2D6 · CYP2C9 · CYP2C19 · CYP1A2 · CYP2E1 · CYP2B6 · UGT · SULT · FMO · AO · MAO · NAT · COMT · GST

---

## Project structure

```
HRMS-Predict/
├── app/
│   ├── main.py          # FastAPI backend
│   ├── frontend.py      # Streamlit UI
│   └── engine/
│       └── metabolism.py  # Core prediction pipeline
├── benchmark/
│   ├── run_metxbiodb.py   # MetXBioDB benchmark script
│   └── results/           # Benchmark output files
├── generate_report.py   # PDF report generator
├── Dockerfile
├── docker-compose.yml
├── environment.yml
└── requirements.txt
```

---

## Citation

If you use HRMS·Predict in your research, please cite:

```bibtex
@software{hegde2026hrmspredict,
  author    = {Hegde, Pooja},
  title     = {{HRMS-Predict: An Open-Source Ensemble Platform for
                In Silico Metabolite Prediction with HRMS Target List Generation}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20518280},
  url       = {https://github.com/PHegde62/HRMS-Predict},
  note      = {RRID:SCR\_028518}
}
```

Or in text: *Hegde P. HRMS-Predict (2026). Genesis Molecular AI. doi:10.5281/zenodo.20518280. RRID:SCR_028518*

---

## Limitations

- Single-step biotransformations only (no sequential cascade prediction)
- Esterase hydrolysis rules not included (SyGMa limitation)
- AO-specific azaheterocycle oxidation partially covered
- Metabolite stereochemistry at new chiral centres not predicted
- Reactive intermediate bioactivation (quinones, arene oxides) not covered
- For research use only — predictions are computational estimates

---

## Roadmap

- [ ] BioTransformer 4.0 integration
- [ ] Aldehyde oxidase expanded rule set
- [ ] Esterase hydrolysis rules for prodrugs
- [ ] Abundance/likelihood ML prediction
- [ ] Multi-step sequential metabolism
- [ ] Species comparison (rat, dog, human)

---

## Contributing

Issues and pull requests are welcome. Please open an issue to discuss proposed changes before submitting a PR.

---

## Licence

Copyright 2026 Pooja Hegde, Genesis Molecular AI.
Licensed under the [Apache License 2.0](LICENSE).

---

<div align="center">
<sub>Built with ❤️ at <a href="https://genesismolecularai.com">Genesis Molecular AI</a></sub>
</div>
