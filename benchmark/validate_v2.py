"""
benchmark/validate_v2.py
========================
Reproducible before/after validation of the v2 metabolism-engine improvements
on the 5 Pharmaron-validated GEN compounds.

WHAT THIS DOES
--------------
For each compound it runs the real `aggregate_metabolism()` engine, then applies
the v1 (git HEAD) and v2 (deployed) `apply_all_improvements()` to the SAME
baseline, and reports:
  - custom N-dealkylation products generated (rank / delta / score)
  - custom reduction products generated (rank / delta / score)
  - glucuronide-like candidates present in the baseline
  - which glucuronides were score-boosted by v1 vs v2

IMPORTANT SCOPE LIMITATION
--------------------------
This harness runs **SyGMa + SMARTCyp only** unless you pass run_biotransformer
/ run_dl. BioTransformer (needs the JAR) and the DL models (MetaTrans /
Meta-Predictor) are often not present, so the ranks/scores here are NOT the
deployed-pipeline recall. To produce the manuscript Section 3.5 recall table,
run with the full pipeline AND the experimental matched-metabolite lists.

CONFIDENTIALITY
---------------
The 5 GEN structures are confidential and are NOT stored in this file. They are
loaded at runtime from a local, git-ignored `gen_compounds.local.json`
(map of GEN-ID -> SMILES). Compounds without a local SMILES are skipped.

HOW TO RUN
----------
    python benchmark/validate_v2.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.engine.metabolism import aggregate_metabolism                 # noqa: E402
import app.engine.metabolism_improvements as v2                        # noqa: E402

# Load the v1 improvements straight from git HEAD so the comparison is faithful
# even if the working-tree file has been replaced by v2.
_v1_src = subprocess.check_output(
    ["git", "-C", str(REPO), "show", "HEAD:app/engine/metabolism_improvements.py"],
    text=True,
)
_v1_tmp = tempfile.NamedTemporaryFile("w", suffix="_imp_v1.py", delete=False)
_v1_tmp.write(_v1_src)
_v1_tmp.close()
_spec = importlib.util.spec_from_file_location("imp_v1", _v1_tmp.name)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

# 5 Pharmaron-validated compounds.  Structures are CONFIDENTIAL — loaded from a
# local, git-ignored file (map of GEN-ID -> SMILES); never committed to VCS.
_GEN_IDS = ("GEN-0042603", "GEN-0042983", "GEN-0066703", "GEN-0069550", "GEN-0070577")
try:
    _S = json.load(open(os.environ.get("GEN_SMILES_JSON", REPO / "gen_compounds.local.json")))
except Exception:
    _S = {}
CASES: dict[str, str] = {gid: _S.get(gid, "") for gid in _GEN_IDS}

# Glucuronide signatures actually emitted by the engine (O-gluc lands at
# +176/+192/+208; SyGMa N-glucuronides land ~+177 — see findings doc).
GLUC_SIGNATURES = (176.032, 177.039, 192.027, 193.034, 208.022, 209.030)


def _gluc_like(delta: float) -> bool:
    return any(abs(delta - t) < 0.06 for t in GLUC_SIGNATURES)


def _tagged(mets: list[dict], tag: str) -> list[tuple]:
    out = []
    for i, m in enumerate(mets):
        if m.get("source_pipeline") == tag:
            out.append((i + 1, round(float(m["delta_mass"]), 3),
                        m["ensemble_score"], m["reaction_label"]))
    return out


def _boosted(mets: list[dict]) -> list[tuple]:
    return [(i + 1, round(float(m["delta_mass"]), 3), m["ensemble_score"])
            for i, m in enumerate(mets) if "UGT boost" in m.get("score_note", "")]


def main(run_biotransformer: bool = False, run_dl: bool = False) -> None:
    print("=" * 78)
    print("HRMS-Predict v2 validation  "
          f"(biotransformer={run_biotransformer}, dl={run_dl})")
    print("=" * 78)
    for name, smi in CASES.items():
        if not smi:
            print(f"\n{name}: SMILES not available locally — skipping")
            continue
        raw = aggregate_metabolism(
            smi, run_sygma=True, run_biotransformer=run_biotransformer,
            run_dl=run_dl, run_smartcyp=True,
        )
        base = raw["metabolites"]
        before = v1.apply_all_improvements(
            copy.deepcopy(base), smi, run_sequential=False,
            run_ndealk=True, verbose=False)
        after = v2.apply_all_improvements(
            copy.deepcopy(base), smi, run_sequential=False,
            run_ndealk=True, run_reduction=True, verbose=False)

        print(f"\n=== {name}  (baseline {len(base)} metabolites) ===")
        print(f"  v2 N-dealk products  : {_tagged(after, 'custom_ndealk')}")
        print(f"  v2 reduction products: {_tagged(after, 'custom_reduction')}")
        gl = sorted({round(float(m['delta_mass']), 3)
                     for m in base if _gluc_like(float(m['delta_mass']))})
        print(f"  glucuronide-like deltas in baseline: {gl}")
        print(f"  glucuronides boosted by v1: {_boosted(before)}")
        print(f"  glucuronides boosted by v2: {len(_boosted(after))} "
              f"(top score {max([b[2] for b in _boosted(after)], default=0)})")


if __name__ == "__main__":
    main()
