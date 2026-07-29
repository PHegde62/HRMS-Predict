# HRMS-Predict v2 — Improvement Validation Findings

**Date:** 2026-06-11
**Scope of this run:** SyGMa + SMARTCyp pipelines only. BioTransformer (no JAR present) and the DL models (`models/metatrans`, `models/metapredictor` are empty) were **disabled**. All 5 GEN compound structures were taken from the v2 self-test; they remain confidential and are referenced by GEN-ID only.
**Reproduce with:** `python benchmark/validate_v2.py`

---

## Summary

v2 was deployed over v1 (`app/engine/metabolism_improvements.py` and `app/main.py`; v1 backed up as `*.v1.bak`, and recoverable from git HEAD). Both files pass `ast.parse`. The 9-case self-test logic was exercised against the **real engine** rather than the synthetic fixtures in the self-test.

Three of the four fixes do what they claim. The glucuronidation fix works only for the oxidative (+192) class in this reduced pipeline, and several products that v2 now *generates* still rank too low to surface in a top-12 report. None of this is visible from the self-test, which uses surrogate SMILES and a score>0.20 pass criterion.

---

## What v2 demonstrably improves vs v1

| Fix | Behaviour confirmed against the real engine | v1 | v2 |
|---|---|---|---|
| **1 — UGT boost (+192 ox-gluc)** | Every compound carries +192.027 ox+glucuronide isomers from the SyGMa phase1→phase2 tree. v2 boosts them to the top of the list. | boosts **0** | boosts to **rank 1**, score 0.50–0.55 |
| **2 — Piperazine N-aryl dealkylation** | GEN-0042983 −80.04 and GEN-0066703 −128.02 products are now generated. | **0 products** | generated (see gaps below for rank) |
| **4 — Carbonyl/lactam reduction (+2)** | GEN-0069550 and GEN-0070577 +2.016 lactam-reduction products now generated. | **0 products** | generated (rank 46–58) |

The headline win is **GEN-0070577**: its +192 ox+glucuronide (the major monkey metabolite) moves from **rank 31 / score 0.15 (v1)** to **rank 1 / score 0.50 (v2)**.

v1 boosts **no** glucuronides on any of the 5 compounds, because v1 only matched Δ+176.032 exactly and the SyGMa tree never produces a clean direct +176 (see gap A).

---

## Gaps found (these matter for Section 3.5 and for the next code pass)

**A. Direct +176 and N-glucuronides are not captured.**
The two-step SyGMa tree applies phase II only to phase I products, never to the parent — so a *direct* parent glucuronide (+176) is never generated. When phase II is applied to the parent directly, SyGMa emits the **N-glucuronide at +177.04**, which is ~1 Da outside v2's ±0.020 boost window. GEN-0069550/0070577 show a +177.016 candidate in the baseline that v2 leaves un-boosted. **Consequence:** GEN-0042603's dominant human metabolites (sulfonamide N-glucuronides, 42.78%) would be missed in a SyGMa-only run. In the deployed pipeline these direct glucuronides most likely come from **BioTransformer**, which could not be run here — so glucuronide recall cannot be validated without the JAR.
*Suggested fix:* widen the glucuronide tolerance to ≈0.05 Da and add a +177.04 target, or generate direct glucuronides explicitly rather than relying on the phase1→phase2 tree.

**B. Custom dealk/reduction products rank too low to appear in a top-12 report.**
N-dealkylation products score a flat 0.28 and reduction products 0.22, which lands them at ranks 30–58 behind the SyGMa oxidation block. GEN-0042983 −80 dealk → rank 42; GEN-0066703 −128 dealk → rank 30; reductions → rank 46–58. They are generated but invisible at `top_n=12`.

**C. The flat-score problem moved to the top rather than being solved.**
All +192 ox-gluc isomers receive the *same* boosted score (0.50/0.55), creating a large tie block at the head of the list (e.g. GEN-0069550 has 35 isomers tied at 0.55). Rank ordering among glucuronide isomers is still arbitrary, and at `top_n=12` the ranked list can be entirely +192 isomers, pushing out other real metabolites. `rescore_cyp_priority` does not differentiate these because they are phase II (UGT), not CYP-labelled.

**D. Noise products from the N-aryl SMIRKS.**
The `[NX3;R1;r6]-[c]` cleavage also yields large spurious fragments (e.g. −336, −365, −407 Da) that pass the heavy-atom filter. Harmless to recall but they add clutter to `return_all` target lists.

**E. Pre-existing engine bug (not v2).**
SMARTCyp rule `2C9_benzylic_near_EWG` = `[CH2]-c1ccc([F,Cl,Br,C(=O)])cc1` fails to compile in RDKit 2026.03 (`C(=O)` inside an atom OR-list is invalid SMARTS) and is silently skipped. Worth fixing independently.

---

## What is still needed for the manuscript Section 3.5 recall table

This run validates the *mechanism* of each fix but **cannot produce the deployed-pipeline recall percentages**, because:

1. **BioTransformer JAR** — absent. Likely the main generator of direct glucuronides; required for honest glucuronidation recall.
2. **DL model checkpoints** — `models/metatrans` and `models/metapredictor` are empty; multi-source ensemble scoring (and therefore ranks) differs without them.
3. **Experimental matched-metabolite lists** — needed to compute matched/total recall per compound (you indicated you will supply these).

Once (1)–(3) are in place, `benchmark/validate_v2.py` can be re-run with `run_biotransformer=True, run_dl=True` to produce the before/after table.
