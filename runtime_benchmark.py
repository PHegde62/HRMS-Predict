"""
Runtime Benchmark for HRMS-Predict — Section 2.7 (Performance)
================================================================
Run with your backend running:
    conda activate hrms-predictor
    uvicorn app.main:app --port 8000 &
    python runtime_benchmark.py

Outputs a table of mean/min/max prediction times for four compounds.
Share the output and it will be written into the manuscript Section 2.7.
"""
import time, requests, json

BASE_URL  = "http://localhost:8000"
PREDICT   = f"{BASE_URL}/predict"
N_REPEATS = 3

compounds = [
    ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O",           180.04,  "Small (~180 Da)"),
    ("Warfarin",      "OC(=O)CCc1ccc(OCC(=O)c2ccccc2)cc1", 308.10, "Medium (~308 Da)"),
    ("Atorvastatin",  "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)"
                      "c(CCC(O)CC(O)CC(=O)O)n1CCc1ccc(F)cc1",
                                                           558.25, "Large (~558 Da)"),
    ("Cyclosporine",  "CCC1NC(=O)C(C(=O)N(C)C(CC(C)C)C(=O)N(C)"
                      "C(CC(C)C)C(=O)NC(C(=O)N(C)C(Cc2ccccc2)C(=O)N"
                      "C(C)C(=O)N(C)C(CC(C)C)C(=O)NC(CCC)C(=O)N1C)C(C)CC)"
                      "C(C)O",                            1202.60, "Complex (cyclosporine)"),
]

print()
print("=" * 72)
print(f"HRMS-Predict Runtime Benchmark  (n={N_REPEATS} repeats, SyGMa+SMARTCyp only)")
print("=" * 72)
print(f"{'Compound':20s} {'MW(Da)':>8s} {'n_pred':>7s} {'Mean(s)':>8s} "
      f"{'Min(s)':>8s} {'Max(s)':>8s}")
print("-" * 72)

for name, smi, mw, size in compounds:
    times, n_pred = [], 0
    for rep in range(N_REPEATS):
        t0 = time.time()
        try:
            r = requests.post(PREDICT, json={
                "smiles": smi,
                "run_sygma": True,
                "run_biotransformer": False,
                "run_dl": False,
                "run_smartcyp": True,
            }, timeout=180)
            elapsed = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                n_pred = data.get("pipeline_stats", {}).get("total_after_dedup", 0)
                times.append(elapsed)
            else:
                print(f"  ERROR {r.status_code}: {r.text[:80]}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")

    if times:
        print(f"{name:20s} {mw:>8.2f} {n_pred:>7d} "
              f"{sum(times)/len(times):>8.2f} "
              f"{min(times):>8.2f} {max(times):>8.2f}")
    else:
        print(f"{name:20s} {mw:>8.2f} {'ERROR':>7s}")

print("=" * 72)
print(f"\nShare this output to have Section 2.7 written.")
