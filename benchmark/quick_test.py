"""
benchmark/quick_test.py
=======================
Runs the benchmark on just 10 compounds to verify the pipeline
works end-to-end before committing to the full MetXBioDB run.

Usage:
    conda activate hrms-predictor
    python benchmark/quick_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors

PREDICT_URL = "http://localhost:8000/predict"
HEALTH_URL  = "http://localhost:8000/health"

# 10 well-characterised compounds with known major metabolites
# Format: (name, smiles, [(metabolite_name, expected_delta_mass)])
TEST_COMPOUNDS = [
    ("Diclofenac",
     "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
     [("4'-Hydroxydiclofenac", +15.9949), ("Diclofenac glucuronide", +176.032)]),

    ("Acetaminophen",
     "CC(=O)Nc1ccc(O)cc1",
     [("Paracetamol sulfate", +79.957), ("Paracetamol glucuronide", +176.032)]),

    ("Caffeine",
     "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
     [("Paraxanthine", -14.016), ("Theobromine", -14.016)]),

    ("Warfarin",
     "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
     [("7-Hydroxywarfarin", +15.9949), ("Warfarin alcohol", +2.016)]),

    ("Dextromethorphan",
     "COc1ccc2c(c1)[C@@H]1CCN(C)[C@@H]1CCC2O",
     [("Dextrorphan", -14.016), ("3-Methoxymorphinan", -14.016)]),

    ("Omeprazole",
     "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
     [("5-Hydroxyomeprazole", +15.9949), ("Omeprazole sulfone", +15.9949)]),

    ("Isoniazid",
     "NNC(=O)c1ccncc1",
     [("Acetylisoniazid", +42.011), ("Isonicotinic acid", -14.027)]),

    ("Ibuprofen",
     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
     [("Hydroxyibuprofen", +15.9949), ("Ibuprofen glucuronide", +176.032)]),

    ("Metoprolol",
     "COCCC(=O)Oc1ccc(CC(O)CNC(C)C)cc1",
     [("O-Demethylmetoprolol", -14.016), ("Alpha-hydroxymetoprolol", +15.9949)]),

    ("Tamoxifen",
     "CC(/C=C/c1ccc(OCCN(CC)CC)cc1)c1ccccc1",
     [("N-Desmethyltamoxifen", -14.016), ("Tamoxifen N-oxide", +15.9949)]),
]

MASS_TOL = 0.010  # ±10 mDa tolerance for delta mass matching


def check_server():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run_prediction(smiles: str) -> list[dict] | None:
    try:
        r = requests.post(PREDICT_URL, json={
            "smiles": smiles,
            "run_sygma": True,
            "run_biotransformer": False,
            "run_dl": False,
            "run_smartcyp": True,
        }, timeout=120)
        if r.status_code == 200:
            return r.json().get("metabolites", [])
        print(f"  HTTP {r.status_code}: {r.text[:100]}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("=" * 60)
    print("HRMS·Predict — Quick Benchmark Test (10 compounds)")
    print("=" * 60)

    if not check_server():
        print("\n✗ Backend not running. Start with:")
        print("  uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    print("✓ Backend online\n")

    results = []
    for name, smiles, expected in TEST_COMPOUNDS:
        print(f"Testing: {name}")
        t0 = time.perf_counter()
        mets = run_prediction(smiles)
        elapsed = round(time.perf_counter() - t0, 1)

        if mets is None:
            print(f"  ✗ Prediction failed\n")
            results.append({"compound": name, "status": "FAIL"})
            continue

        # Check each expected metabolite
        found_count = 0
        for met_name, exp_delta in expected:
            # Match by delta mass within tolerance
            hit = any(
                abs(float(m.get("delta_mass", "0").replace("+","").replace("-",""))
                    - abs(exp_delta)) < MASS_TOL
                for m in mets
            )
            status = "✓" if hit else "✗"
            if hit:
                found_count += 1
            rank_info = ""
            if hit:
                # Find the matching rank
                for m in mets:
                    dm = abs(float(m.get("delta_mass","0").replace("+","").replace("-","")))
                    if abs(dm - abs(exp_delta)) < MASS_TOL:
                        rank_info = f" (rank {m.get('rank','?')})"
                        break
            print(f"  {status} {met_name} (Δ{exp_delta:+.4f}){rank_info}")

        print(f"  → {found_count}/{len(expected)} expected metabolites found"
              f"  |  {len(mets)} total predicted  |  {elapsed}s\n")

        results.append({
            "compound": name,
            "expected": len(expected),
            "found": found_count,
            "total_predicted": len(mets),
            "elapsed_s": elapsed,
        })

    # Summary
    print("=" * 60)
    valid = [r for r in results if "found" in r]
    total_expected = sum(r["expected"] for r in valid)
    total_found    = sum(r["found"]    for r in valid)
    print(f"Overall: {total_found}/{total_expected} expected metabolites found "
          f"({total_found/total_expected*100:.0f}% recall)")
    print(f"Mean prediction time: {sum(r['elapsed_s'] for r in valid)/len(valid):.1f}s")
    print("=" * 60)

    # Save results
    Path("benchmark/results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv("benchmark/results/quick_test_results.csv", index=False)
    print("\nResults saved to benchmark/results/quick_test_results.csv")


if __name__ == "__main__":
    main()
