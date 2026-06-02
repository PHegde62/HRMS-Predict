"""
benchmark/run_metxbiodb.py
==========================
MetXBioDB Benchmark Runner for HRMS·Predict

Runs every unique substrate in MetXBioDB through the HRMS·Predict
/predict endpoint, then computes recall and precision metrics by
matching predicted metabolites against known metabolites using
InChIKey or exact mass (±5 ppm) as matching criteria.

Usage
-----
    conda activate hrms-predictor
    # Make sure both servers are running, then:
    python benchmark/run_metxbiodb.py

Outputs
-------
    benchmark/results/metxbiodb_results.csv   — per-compound results
    benchmark/results/metxbiodb_summary.csv   — aggregate metrics
    benchmark/results/metxbiodb_report.txt    — human-readable summary

The script is resumable: if metxbiodb_results.csv already exists,
compounds already processed are skipped.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METXBIODB_CSV   = Path("data/metxbiodb/metxbiodb.csv")
RESULTS_DIR     = Path("benchmark/results")
RESULTS_CSV     = RESULTS_DIR / "metxbiodb_results.csv"
SUMMARY_CSV     = RESULTS_DIR / "metxbiodb_summary.csv"
REPORT_TXT      = RESULTS_DIR / "metxbiodb_report.txt"

PREDICT_URL     = "http://localhost:8000/predict"
HEALTH_URL      = "http://localhost:8000/health"

# Only run Human Phase I and Phase II biotransformations
# (matches what SyGMa + SMARTCyp cover)
BIOSYSTEM_FILTER = {"Human", "Human Phase I", "Human Phase II"}

# Mass matching tolerance in Da
# (5 ppm at 300 Da ≈ 0.0015 Da; we use 0.005 Da to be generous)
MASS_TOLERANCE_DA = 0.005

# Max compounds to process (None = all)
# Set to a number e.g. 50 for a quick test run
MAX_COMPOUNDS = None

# Request timeout
TIMEOUT_S = 120

# Delay between API calls (seconds) — be gentle to the local server
DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------

def inchi_to_smiles(inchi: str) -> str | None:
    """Convert InChI string to canonical SMILES via RDKit."""
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_to_mass(smiles: str) -> float | None:
    """Return monoisotopic neutral mass from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return round(Descriptors.ExactMolWt(mol), 4)
    except Exception:
        return None


def smiles_to_inchikey(smiles: str) -> str | None:
    """Return InChIKey from SMILES, None if unavailable."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = Chem.MolToInchi(mol)
        if inchi:
            return rdMolDescriptors.CalcInchiKey(inchi)
        # Fallback: first 20 chars of SHA256 of canonical SMILES
        import hashlib
        smi = Chem.MolToSmiles(mol)
        return "SMIKEY-" + hashlib.sha256(smi.encode()).hexdigest()[:20]
    except Exception:
        return None


def masses_match(m1: float, m2: float, tol: float = MASS_TOLERANCE_DA) -> bool:
    return abs(m1 - m2) <= tol


# ---------------------------------------------------------------------------
# Load and preprocess MetXBioDB
# ---------------------------------------------------------------------------

def load_metxbiodb(path: Path) -> pd.DataFrame:
    """
    Load MetXBioDB CSV and filter to human biotransformations.
    Returns a DataFrame with one row per substrate–product pair,
    with added SMILES columns for both.
    """
    print(f"Loading MetXBioDB from {path} …")
    df = pd.read_csv(path, quotechar='"')
    print(f"  Total rows: {len(df)}")

    # Filter to human biosystems
    df = df[df["biosystem"].isin(BIOSYSTEM_FILTER)].copy()
    print(f"  After human filter: {len(df)} rows")

    # Convert substrate InChI → SMILES
    print("  Converting substrate InChI → SMILES …")
    df["substrate_smiles"] = df["substrate_inchi"].apply(inchi_to_smiles)

    # Convert product InChI → SMILES
    print("  Converting product InChI → SMILES …")
    df["prod_smiles"] = df["prod_inchi"].apply(inchi_to_smiles)

    # Drop rows where either conversion failed
    before = len(df)
    df = df.dropna(subset=["substrate_smiles", "prod_smiles"]).copy()
    print(f"  After SMILES conversion: {len(df)} rows ({before - len(df)} dropped)")

    # Compute product masses
    df["prod_mass"] = df["prod_smiles"].apply(smiles_to_mass)
    df["prod_inchikey_computed"] = df["prod_smiles"].apply(smiles_to_inchikey)

    return df


def build_substrate_index(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Group MetXBioDB rows by substrate InChIKey.
    Returns {substrate_inchikey: [list of known metabolite dicts]}.
    """
    index: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        key = row["substrate_inchikey"]
        if key not in index:
            index[key] = []
        index[key].append({
            "prod_name":     row.get("prod_name", ""),
            "prod_smiles":   row["prod_smiles"],
            "prod_mass":     row["prod_mass"],
            "prod_inchikey": row["prod_inchikey_computed"],
            "enzyme":        row.get("enzyme", ""),
            "reaction_type": row.get("reaction_type", ""),
            "phase":         row.get("biotransformation_type", ""),
        })
    return index


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def find_matches(
    predicted: list[dict[str, Any]],
    known: list[dict],
) -> tuple[int, list[dict]]:
    """
    For each known metabolite, check whether it appears anywhere
    in the predicted list using two criteria (in priority order):

    1. InChIKey match (exact structural identity)
    2. Neutral mass match within ±MASS_TOLERANCE_DA

    Returns (n_matched, match_detail_list).
    """
    matched = 0
    details = []

    for km in known:
        hit = False
        hit_rank = None
        hit_method = ""

        for pred in predicted:
            # Criterion 1: InChIKey
            p_ik = pred.get("inchikey", "")
            k_ik = km.get("prod_inchikey", "")
            if p_ik and k_ik and p_ik == k_ik:
                hit       = True
                hit_rank  = pred.get("rank", 0)
                hit_method = "InChIKey"
                break

            # Criterion 2: Mass within tolerance
            p_mass = pred.get("neutral_mass", 0.0)
            k_mass = km.get("prod_mass") or 0.0
            if k_mass and masses_match(float(p_mass), float(k_mass)):
                hit       = True
                hit_rank  = pred.get("rank", 0)
                hit_method = "Mass (±5 ppm)"
                break

        if hit:
            matched += 1

        details.append({
            "prod_name":     km["prod_name"],
            "prod_mass":     km["prod_mass"],
            "enzyme":        km["enzyme"],
            "reaction_type": km["reaction_type"],
            "matched":       hit,
            "rank":          hit_rank,
            "method":        hit_method,
        })

    return matched, details


def recall_at_n(predicted: list[dict], known: list[dict], n: int) -> float:
    """Recall when only considering the top-n predictions."""
    top_n = [p for p in predicted if p.get("rank", 0) <= n]
    matched, _ = find_matches(top_n, known)
    return matched / len(known) if known else 0.0


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def predict(smiles: str) -> list[dict] | None:
    """Call /predict and return the metabolites list, or None on failure."""
    payload = {
        "smiles":            smiles,
        "run_sygma":         True,
        "run_biotransformer": False,
        "run_dl":            False,
        "run_smartcyp":      True,
    }
    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=TIMEOUT_S)
        if r.status_code == 200:
            return r.json().get("metabolites", [])
        print(f"    HTTP {r.status_code}: {r.text[:120]}")
        return None
    except requests.exceptions.Timeout:
        print("    TIMEOUT")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    REQUEST ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(df: pd.DataFrame, substrate_index: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check server health
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        assert r.status_code == 200
        print("✓ Backend is online\n")
    except Exception:
        print("✗ Backend is not responding at localhost:8000")
        print("  Start it with: uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    # Load already-processed compounds if resuming
    done: set[str] = set()
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        done = set(existing["substrate_inchikey"].tolist())
        print(f"Resuming — {len(done)} compounds already processed\n")

    # Unique substrates
    substrates = list(substrate_index.keys())
    if MAX_COMPOUNDS:
        substrates = substrates[:MAX_COMPOUNDS]

    total = len(substrates)
    print(f"Running benchmark on {total} unique substrates …\n")

    # Open results file for appending
    write_header = not RESULTS_CSV.exists()
    results_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(results_file, fieldnames=[
        "substrate_inchikey", "substrate_name", "substrate_smiles",
        "n_known_metabolites",
        "n_predicted",
        "n_matched",
        "recall",
        "recall_top1", "recall_top3", "recall_top5", "recall_top10",
        "precision_top10",
        "elapsed_s",
        "error",
    ])
    if write_header:
        writer.writeheader()

    # Per-compound loop
    for i, ik in enumerate(substrates, 1):
        if ik in done:
            continue

        known = substrate_index[ik]
        sub_name = known[0].get("prod_name", ik)  # use first product row for name

        # Get substrate SMILES from the dataframe
        rows = df[df["substrate_inchikey"] == ik]
        if rows.empty:
            continue
        sub_smiles = rows.iloc[0]["substrate_smiles"]
        sub_name   = rows.iloc[0]["substrate_name"]

        print(f"[{i:4d}/{total}] {sub_name:30s} ({len(known)} known metabolites) …", end=" ", flush=True)

        t0 = time.perf_counter()
        predicted = predict(sub_smiles)
        elapsed   = round(time.perf_counter() - t0, 2)

        if predicted is None:
            print("ERROR")
            writer.writerow({
                "substrate_inchikey": ik,
                "substrate_name":     sub_name,
                "substrate_smiles":   sub_smiles,
                "n_known_metabolites":len(known),
                "n_predicted":        0,
                "n_matched":          0,
                "recall":             0,
                "recall_top1":        0, "recall_top3": 0,
                "recall_top5":        0, "recall_top10": 0,
                "precision_top10":    0,
                "elapsed_s":          elapsed,
                "error":              "API_FAILURE",
            })
            results_file.flush()
            time.sleep(DELAY_S)
            continue

        n_known  = len(known)
        n_pred   = len(predicted)
        n_match, _ = find_matches(predicted, known)

        recall       = round(n_match / n_known, 4) if n_known else 0
        recall_top1  = round(recall_at_n(predicted, known, 1),  4)
        recall_top3  = round(recall_at_n(predicted, known, 3),  4)
        recall_top5  = round(recall_at_n(predicted, known, 5),  4)
        recall_top10 = round(recall_at_n(predicted, known, 10), 4)

        # Precision@10: fraction of top-10 predictions that are real
        top10_preds    = [p for p in predicted if p.get("rank", 0) <= 10]
        true_positives = sum(
            1 for p in top10_preds
            if any(
                masses_match(p.get("neutral_mass", 0), km.get("prod_mass") or 0)
                for km in known
            )
        )
        precision_top10 = round(true_positives / len(top10_preds), 4) if top10_preds else 0

        print(f"recall={recall:.2f}  predicted={n_pred:3d}  matched={n_match}/{n_known}  {elapsed:.1f}s")

        writer.writerow({
            "substrate_inchikey": ik,
            "substrate_name":     sub_name,
            "substrate_smiles":   sub_smiles,
            "n_known_metabolites":n_known,
            "n_predicted":        n_pred,
            "n_matched":          n_match,
            "recall":             recall,
            "recall_top1":        recall_top1,
            "recall_top3":        recall_top3,
            "recall_top5":        recall_top5,
            "recall_top10":       recall_top10,
            "precision_top10":    precision_top10,
            "elapsed_s":          elapsed,
            "error":              "",
        })
        results_file.flush()
        time.sleep(DELAY_S)

    results_file.close()
    print(f"\nBenchmark complete. Results saved to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def compute_summary() -> None:
    if not RESULTS_CSV.exists():
        print("No results file found.")
        return

    df = pd.read_csv(RESULTS_CSV)
    df = df[df["error"] == ""]  # exclude failures

    n = len(df)
    if n == 0:
        print("No valid results to summarise.")
        return

    summary = {
        "n_compounds_tested":       n,
        "mean_recall_overall":      round(df["recall"].mean(), 4),
        "mean_recall_top1":         round(df["recall_top1"].mean(), 4),
        "mean_recall_top3":         round(df["recall_top3"].mean(), 4),
        "mean_recall_top5":         round(df["recall_top5"].mean(), 4),
        "mean_recall_top10":        round(df["recall_top10"].mean(), 4),
        "mean_precision_top10":     round(df["precision_top10"].mean(), 4),
        "median_n_predicted":       int(df["n_predicted"].median()),
        "mean_elapsed_s":           round(df["elapsed_s"].mean(), 2),
        "total_known_metabolites":  int(df["n_known_metabolites"].sum()),
        "total_matched":            int(df["n_matched"].sum()),
        "overall_pooled_recall":    round(
            df["n_matched"].sum() / df["n_known_metabolites"].sum(), 4
        ),
    }

    # Save summary CSV
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)

    # Write human-readable report
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║         HRMS·Predict — MetXBioDB Benchmark Report               ║
╚══════════════════════════════════════════════════════════════════╝

Dataset : MetXBioDB (Zenodo 4085087) — Human Phase I/II
Pipelines: SyGMa Phase I/II + SMARTCyp (non-CYP soft spots)
Matching : InChIKey (exact) OR neutral mass ±5 ppm

─────────────────────────────────────────────────────────────────
Compounds tested        : {summary['n_compounds_tested']}
Total known metabolites : {summary['total_known_metabolites']}
Total matched           : {summary['total_matched']}

─────────────────────────────────────────────────────────────────
RECALL (fraction of known metabolites found)
  Pooled recall (all ranks)  : {summary['overall_pooled_recall']:.1%}
  Mean recall per compound   : {summary['mean_recall_overall']:.1%}
  Mean recall @ top-1        : {summary['mean_recall_top1']:.1%}
  Mean recall @ top-3        : {summary['mean_recall_top3']:.1%}
  Mean recall @ top-5        : {summary['mean_recall_top5']:.1%}
  Mean recall @ top-10       : {summary['mean_recall_top10']:.1%}

─────────────────────────────────────────────────────────────────
PRECISION
  Mean precision @ top-10    : {summary['mean_precision_top10']:.1%}
  Median predicted per cpd   : {summary['median_n_predicted']}

─────────────────────────────────────────────────────────────────
RUNTIME
  Mean prediction time       : {summary['mean_elapsed_s']:.1f} s/compound

─────────────────────────────────────────────────────────────────
LITERATURE COMPARISONS (SyGMa + SMARTCyp baseline)
  SyGMa top-1 recall (Rydberg 2010)  : 67%
  SyGMa top-10 recall (Rydberg 2010) : 90%
  GLORYx overall recall (Tyzack 2021): 77% (non-CYP only)

─────────────────────────────────────────────────────────────────
"""

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Summary saved to {SUMMARY_CSV}")
    print(f"Report saved to  {REPORT_TXT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not METXBIODB_CSV.exists():
        print(f"MetXBioDB CSV not found at {METXBIODB_CSV}")
        print("Download it with:")
        print('  Invoke-WebRequest -Uri "https://zenodo.org/records/4085087/files/metxbiodb.csv" -OutFile "data\\metxbiodb\\metxbiodb.csv"')
        sys.exit(1)

    df     = load_metxbiodb(METXBIODB_CSV)
    index  = build_substrate_index(df)

    print(f"\nUnique substrates: {len(index)}")
    print(f"Biosystems included: {BIOSYSTEM_FILTER}\n")

    run_benchmark(df, index)
    compute_summary()
