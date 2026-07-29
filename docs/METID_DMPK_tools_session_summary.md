# MetID App — Build Session Summary (from parallel `DMPK_tools` session)

> Captured into the HRMS-Predict Cowork session on 2026-07-21 for consolidation.
> This documents work done in a **separate** chat session on the
> `PHegde62/DMPK_tools` repo (`metid_app/`), which is architecturally different
> from this session's `hrms-predict` app (Streamlit). See the reconciliation
> note at the bottom.

## What Was Built
A full-stack Metabolite Identification (MetID) & Soft Spot Analysis web
application for DMPK workflows, combining rule-based chemistry with deep
learning hooks.

## Architecture
```
PHegde62/DMPK_tools (GitHub repo)
├── metid_app/                    ← FastAPI backend
│   ├── app/
│   │   ├── main.py               ← FastAPI server (serves frontend at /)
│   │   ├── engine/
│   │   │   └── metabolism.py     ← Ensemble Consensus Engine v2.1
│   │   └── static/
│   │       └── index.html        ← Frontend (copy from docs/metid/)
│   ├── requirements.txt          ← Render pip deploy
│   ├── .python-version           ← Python 3.11.9
│   └── environment.yml           ← Local conda environment
└── docs/
    ├── index.html                ← DMPK Tools Hub (GitHub Pages)
    └── metid/
        └── index.html            ← MetID frontend (GitHub Pages)
```

## Live URLs
| Service | URL |
|---|---|
| DMPK Hub | https://PHegde62.github.io/DMPK_tools/ |
| MetID App (cloud) | https://PHegde62.github.io/DMPK_tools/metid/ |
| API (Render) | https://metid-app.onrender.com |
| Local API | http://localhost:8000 |
| Local Frontend | http://localhost:8000/ |

## Key Features
- **Dual pipeline:** SyGMa rule-based (Phase I/II) + DeepLearningPredictor emulator (MetaTrans/Meta-Predictor architecture)
- **Consensus engine:** tags metabolites as High Confidence (both pipelines agree), Rule-Only, or DL-Only
- **LC-MS mass tracking:** exact monoisotopic masses, Δm/z shifts, transformation type annotation (25 curated transformations)
- **Soft spot analysis:** SMARTS-based vulnerability scoring + DL attention weights → Vulnerability Index (%)
- **3-panel dashboard:** Analytics | Soft Spot SVG map | Metabolic Tree with Δm/z table

## Mass Spec Fields (per metabolite)
| Field | Example |
|---|---|
| exact_mass | 196.0372 Da |
| parent_exact_mass | 180.0423 Da |
| mass_shift | +15.9949 |
| mass_shift_str | "+15.9949" |
| transformation_type | "Mono-hydroxylation (+O)" |
| molecular_formula | C9H8O5 |
| vulnerable_atom_idx | 7 |
| confidence_tier | "High Confidence (Consensus Verified)" |

## Local Setup (Full SyGMa Pipeline)
Prerequisites: Anaconda, Git.

Steps (Anaconda Prompt):
```
conda env remove -n metid -y
cd C:\Users\pooja_genesistherape\DMPK_tools\metid_app
conda env create -f environment.yml
conda activate metid
pip install "setuptools<70" --force-reinstall
pip install sygma --no-build-isolation
mkdir app\static
copy ..\docs\metid\index.html app\static\index.html
& "C:\Users\pooja_genesistherape\.conda\envs\metid\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000
```
Open: http://localhost:8000/

### Known fix required after conda install
SyGMa 1.1.0 uses dict syntax — fix in `metabolism.py`:
`srs.phase1 → srs["phase1"]`, `srs.phase2 → srs["phase2"]`
```powershell
$file = "C:\Users\pooja_genesistherape\DMPK_tools\metid_app\app\engine\metabolism.py"
(Get-Content $file) -replace 'srs\.phase1','srs["phase1"]' -replace 'srs\.phase2','srs["phase2"]' | Set-Content $file
```

## Cloud Deployment (Render — free tier)
- Limitation: SyGMa cannot install on Render free tier (conda-only). DL emulator pipeline runs only.
- Fix: Upgrade Render to paid Docker tier ($7/mo) for full SyGMa support.
- Auto-deploys on every `git push origin main` via GitHub Actions (`.github/workflows/deploy.yml`).

## GitHub Actions CI/CD
`.github/workflows/deploy.yml` runs on push to main: black, ruff, pytest, Docker build + smoke test, deploy to Render (via deploy hook secret).
Required secret: `RENDER_DEPLOY_HOOK`.

## Files Modified (that session)
| File | Change |
|---|---|
| app/engine/metabolism.py | Added exact mass, mass_shift, transformation_type, _MASS_SHIFT_ANNOTATIONS (25 transformations), EnsembleSoftSpot, DeepLearningPredictor |
| app/main.py | Lightweight FastAPI, serves static frontend at /, full traceback error reporting |
| docs/metid/index.html | LC-MS mass tracking table, Δm/z colour tags, group-by-shift sorting, parent mass bar |
| docs/index.html | DMPK Tools Hub with MetID card |
| metid_app/requirements.txt | Render pip deploy (Python 3.11, no pandas/conda deps) |
| metid_app/.python-version | Pins Python 3.11.9 for Render |
| environment.yml | Conda env without SyGMa (install separately) |

## Next Steps / To-Do (that session)
- [ ] Install SyGMa locally and confirm full Phase I/II metabolite predictions run
- [ ] Upgrade Render to paid tier for cloud SyGMa support
- [ ] Swap DeepLearningPredictor emulator for real MetaTrans/Meta-Predictor .pt weights
- [ ] Add streamlit-ketcher molecule sketcher to frontend
- [ ] Add export to CSV/Excel for LC-MS correlation workflow

---

## Reconciliation note (HRMS-Predict / this session)
The `hrms-predict` app already implements a superset of the chemistry engine
(dual-track glucuronidation, scaffold fragmentation, DMPK-empirical
prioritization, SMARTCyp soft-spot map, CDD Vault GEN-ID lookup, ReportLab PDF).
Features from the `DMPK_tools` session to reconcile into this codebase are the
**DL-emulator consensus tiering** (High Confidence / Rule-Only / DL-Only) and
the explicit **molecular_formula** field. Mass tracking, Δm/z, transformation
annotations, and soft-spot vulnerability scoring already exist here in
equivalent form. See audit below / chat for the gap analysis.

---
---

# HRMS-Predict — Full Project Handoff (Sessions 1 + 2)

## What this project is
HRMS-Predict is an open-source ensemble metabolite prediction platform built by
Pooja Hegde at Genesis Molecular AI. Predicts Phase I and Phase II drug
metabolites, generates HRMS-grade exact masses ([M+H]⁺ and [M−H]⁻ to 4 d.p.),
and produces enzyme-annotated soft-spot visualisations. Runs entirely locally.

- **GitHub:** https://github.com/PHegde62/HRMS-Predict (private, awaiting C-suite approval)
- **Zenodo DOI:** 10.5281/zenodo.20518280 (restricted)
- **RRID:** SCR_028518
- **Licence:** Apache 2.0 — Copyright 2026 Pooja Hegde, Genesis Molecular AI
- **Local path:** `C:\Users\pooja_genesistherape\hrms-predict\`
- **Conda environment:** `hrms-predictor`

### Startup commands
```powershell
cd C:\Users\pooja_genesistherape\hrms-predict
conda activate hrms-predictor
python -m uvicorn app.main:app --reload --port 8000
# Second window for Streamlit:
streamlit run app/frontend.py
```

## Manuscript status
- **File:** `hrms_predict_manuscript_v12.docx`
- **Target journal:** Journal of Cheminformatics
- **Status:** Complete except **Section 3.5 (experimental validation)** — not yet written.
- All other sections done including Discussion, Limitations, Table 2 (8-tool comparison), F1=0.305 at rank 10, Reference 19 (2026 JCIM benchmark), RRID SCR_028518.

## Experimental validation — Pharmaron LC-HRMS data (5 compounds)
Pharmaron PPTX files are password-protected (password: `phges`). Files named by
GEN-ID. Structures confidential — referred to by GEN-ID only.

- **Overall recall before improvements:** 35.9% (14/39); human-relevant: 37.5% (12/32)

| Compound | Matrix | Exp. mets | Matched | Recall |
|---|---|---|---|---|
| GEN-0042603 | Hepatocytes (rat/dog/human) | 9 | 3 | 33% |
| GEN-0042983 | Hepatocytes (rat/dog/human) | 10 | 4 | 40% |
| GEN-0066703 | Human liver microsomes | 5 | 4 | 80% |
| GEN-0069550 | Rat plasma (in vivo PK) | 4 | 2 | 50% |
| GEN-0070577 | Hepatocytes (rat/dog/monkey/human) | 11 | 2 | 18% |

### Four systematic failure modes
1. **Glucuronidation (8 misses)** — UGT metabolites scored too low; worst for GEN-0042603 (4 glucuronide isomers = 42.78% human peak area) and GEN-0069550 (M2 = 14.48%).
2. **Large piperazine N-dealkylation (5 misses)** — GEN-0042983 M3 (−80 Da, 42% rat) and GEN-0066703 M4 (−128 Da); v1 SMIRKS produced zero products due to SMARTS matching bug.
3. **Sequential/bi-oxidation (5 misses)** — tool doesn't chain Phase I → Phase II; GEN-0070577 M4 is Ox+Gluc (+192 Da).
4. **Reduction/hydrogenation (3 misses)** — carbonyl reduction (+2 Da) absent; GEN-0069550 M4 and GEN-0070577 M8.
- **Flat 0.200 scoring** — all sygma-only predictions scored identically; rank ordering meaningless.

- **Estimated recall after improvements:** ~55–60% overall

## Code changes — current deployed state
Two files updated. Both deployed to local machine and confirmed working.

### `app/engine/metabolism_improvements.py` — v2
**Fix 1+2 — UGT boost extended:** Covers Δ+176 (direct gluc), Δ+192 (ox+gluc),
Δ+208 (bi-ox+gluc), Δ+178 (hydrog+gluc). Previously only Δ+176.

UGT site boosts as currently deployed:

| Site | SMARTS | Boost |
|---|---|---|
| sulfonamide_NH2 | `[S](=[O])(=[O])[NH2]` | 0.40 |
| sulfonamide_NH | `[S](=[O])(=[O])[NH]` | 0.35 |
| aromatic_NH2 | `[NH2]c` | 0.30 |
| NH_heteroaromatic | `[nH]` | 0.35 |
| pyrazolone_NH | `[nH]n` | 0.40 |
| aliphatic_OH | `[OHX2][CX4]` | 0.35 |
| phenol | `[OH]c` | 0.30 |
| carboxylic_acid | `[CX3](=[OX1])[OHX2]` | 0.10 (reduced from 0.45 — see bug fix below) |
| hydroxamic_acid | `[NH][OH]` | 0.30 |

**Fix 2 — N-dealkylation SMIRKS replaced:** v1 produced zero products. v2 uses
`[NX3;R1;r6:1]-[c:2]>>[NH1;R1;r6:1]` which correctly cleaves N-aryl piperazine
bonds. GEN-0042983 M3 (−80 Da) and GEN-0066703 M4 (−128 Da) now generated at score 0.35.

**Fix 3 — CYP structural rescorer:** Breaks flat 0.200 baseline. Benzylic C-H,
N-methyl on aromatic ring, alpha-N-in-ring get up to +0.20 bonus. CHF2 arene gets
−0.05 (metabolically stable).

**Fix 4 — Carbonyl/N-oxide reduction:** New SMIRKS for ketone→alcohol, lactam
carbonyl, N-oxide reduction. Generates +2 Da products. Covers GEN-0069550 M4 and
GEN-0070577 M8. Score set to 0.220.

Self-test: `python app/engine/metabolism_improvements.py` → all 9/9 pass.

### `app/main.py` — 6 patches, backward-compatible
New optional fields in `PredictRequest` (all default to original behaviour):
```json
{
  "top_n": 12,            // ranked metabolites shown (range 1–500, default 12)
  "return_all": false,    // populates all_metabolites in response if true
  "run_sequential": false,
  "run_ndealk": true,
  "run_reduction": true
}
```
New field in `PredictResponse`: `all_metabolites: list[MetaboliteEntry]` — full
list, same schema as `metabolites`, empty unless `return_all: true`.

### Bug found and fixed in Session 2
- **Problem:** Diclofenac (standard: `OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl`) showed all top 5 metabolites as Glucuronidation (score=0.60). Real dominant metabolites are CYP hydroxylations (4'-OH, 5-OH).
- **Root cause:** Carboxylic acid UGT boost (0.45) was too aggressive — fired on diclofenac's carboxylic acid and pushed all glucuronide isomers above CYP metabolites.
- **Fix:** Reduced carboxylic acid boost from 0.45 → 0.10. All other boosts unchanged. Carboxylic acids form acyl glucuronides but these are often minor/reactive — the strong signals that matter for the 5 GEN compounds are sulfonamide-NH (0.40) and pyrazolone-NH (0.40).

Diagnostic script saved at `C:\Users\pooja_genesistherape\hrms-predict\diag.py`:
```powershell
python diag.py
```
Expected healthy output for diclofenac: Rank 1–5 = hydroxylation/aromatic
hydroxylation, score ~0.20–0.35, enzyme CYP. Glucuronidation should not appear in top 5.

## Pending work
| Priority | Task | Status |
|---|---|---|
| 1 | Run all 5 GEN compounds through updated tool — measure before/after recall | Not done |
| 2 | Write Section 3.5 — needs Priority 1 results | Not done |
| 3 | Add variable top_n + full-list appendix to generate_report.py | generate_report.py not yet uploaded |
| 4 | Test run_sequential=True on GEN-0070577 for bi-oxidation+gluc | Not done |
| 5 | C-suite approval for GitHub repo + Zenodo embargo lift | Pending with Indra |
| 6 | GitHub Pages HTML demo | After repo goes public |
| 7 | Export Figures 1–7 as PNG for manuscript | Not done |
| 8 | Paste Table 3 + Table 4 into manuscript v12 | Files exist, manual paste needed |

## Files still needed in next session
- `generate_report.py` — for PDF top_n and full-list appendix
- `app/frontend.py` — if Streamlit UI changes needed
- `app/engine/metabolism.py` — if core engine changes needed
- `hrms_predict_manuscript_v12.docx` — for writing Section 3.5
