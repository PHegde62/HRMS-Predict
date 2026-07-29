"""
app/engine/metabolism.py
========================
HRMS Predictor — Core Metabolism Processing Engine

Aggregates five prediction paradigms:
  Pipeline A  — SyGMa Phase I / II rule-based metabolite tree
  Pipeline B  — BioTransformer JAR subprocess wrapper
  Pipeline C  — MetaTrans / Meta-Predictor DL transformer inference
  Pipeline D  — SMARTCyp P450 fragment-based soft-spot profiler
  Mass Tracker — monoisotopic mass, delta mass, [M+H]+, [M-H]-

All pipelines return a normalised MetaboliteRecord; the final
aggregate_metabolism() call merges, deduplicates on InChIKey, and
returns a structured output matrix.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch  # optional: only used by the DL (MetaTrans/Meta-Predictor) pipeline
except Exception:  # pragma: no cover - heavy optional dependency
    torch = None
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# SyGMa is installed from git+https://github.com/3D-e-Chem/sygma.git
import sygma

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTON_MASS: float = 1.007276          # Da — unified atomic mass of H+
MASS_PRECISION: int = 4                # decimal places for all mass fields

# SMARTCyp P450 site-of-metabolism SMARTS patterns with DFT-derived
# activation energy thresholds (kcal/mol).  Patterns are drawn from the
# published SMARTCyp 2.4.2 rule set (Rydberg et al. 2010, J. Chem. Inf.
# Model.) and cover the three major P450 isoforms (3A4, 2D6, 2C9).
# Each tuple is: (pattern_label, SMARTS, Ea_kcal_per_mol, isoform_hint).
SMARTCYP_RULES: list[tuple[str, str, float, str]] = [
    # ══════════════════════════════════════════════════════════════════════
    # CYP-mediated reactions (SMARTCyp 2.4.2 rule set)
    # Ea values in kcal/mol — lower = more reactive
    # ══════════════════════════════════════════════════════════════════════

    # ── CYP3A4 aliphatic / benzylic hydroxylation ──
    ("3A4_sp3_CH_alpha_N",       "[CX4;H1,H2;!$(CC=O)]-[NX3]",               86.1, "CYP3A4"),
    ("3A4_benzylic_CH",          "[CH2,CH1;!$(CC=O)]-c",                       88.4, "CYP3A4"),
    ("3A4_allylic_CH",           "[CH2,CH1]-[CH]=[CH]",                        89.2, "CYP3A4"),
    ("3A4_tertiary_sp3",         "[CX4;H1](-[CX4])(-[CX4])-[CX4]",            90.5, "CYP3A4"),
    ("3A4_secondary_sp3",        "[CX4;H2](-[CX4])-[CX4]",                    92.0, "CYP3A4"),
    # ── CYP3A4 aromatic hydroxylation ──
    ("3A4_aromatic_C_unsubst",   "c[H]",                                       94.0, "CYP3A4"),
    ("3A4_electron_rich_arene",  "c1ccc([OH1,NH2,OC])cc1",                     87.6, "CYP3A4"),
    # ── CYP3A4 N-dealkylation / N-oxidation ──
    ("3A4_N_CH3",                "[NX3;!$(N-C=O)]-[CH3]",                      84.3, "CYP3A4"),
    ("3A4_N_CH2",                "[NX3;!$(N-C=O)]-[CH2]-[CX4]",               85.7, "CYP3A4"),
    ("3A4_N_oxidation",          "[NX3;H0;!$(N-C=O);!$(N~[!#6])]",            91.1, "CYP3A4"),
    # ── CYP3A4 O-dealkylation ──
    ("3A4_O_CH3",                "[OX2;!$(OC=O)]-[CH3]",                       83.8, "CYP3A4"),
    ("3A4_O_CH2",                "[OX2;!$(OC=O)]-[CH2]-[CX4]",                85.0, "CYP3A4"),
    # ── CYP3A4 S-oxidation ──
    ("3A4_S_oxidation",          "[SX2;H0;!$(S~[!#6])]",                       78.5, "CYP3A4"),
    # ── CYP2D6 preferred substrates ──
    ("2D6_basic_N_CH",           "[NX3;!$(N-C=O);!$(N~[!#6])]-[CX4;H1,H2]",  83.2, "CYP2D6"),
    ("2D6_aromatic_C_ortho_N",   "c-[CX4]-[NX3]",                             85.8, "CYP2D6"),
    ("2D6_phenethyl_CH",         "c-[CX4;H2]-[CX4;H2]-[NX3]",                84.6, "CYP2D6"),
    ("2D6_indole_C2",            "c1cc2ccccc2[nH]1",                           88.9, "CYP2D6"),
    # ── CYP2C9 preferred substrates ──
    ("2C9_acidic_drug_CH",       "[CX4;H1,H2]-[CX3](=O)-[OX1,OX2H]",         89.3, "CYP2C9"),
    ("2C9_benzylic_near_EWG",    "[CH2]-c1ccc([F,Cl,Br,C(=O)])cc1",           90.7, "CYP2C9"),
    ("2C9_methylenedioxy",       "[CH2]1OC(c2ccccc2)O1",                       82.1, "CYP2C9"),

    # ══════════════════════════════════════════════════════════════════════
    # UGT (UDP-Glucuronosyltransferase) — Phase II conjugation sites
    # Ea proxy: phenols < primary amines < secondary amines < carboxylic acids
    # Reference: Miners & Mackenzie, Pharmacol. Ther. 1991; Soars et al. 2004
    # ══════════════════════════════════════════════════════════════════════
    ("UGT_phenol_O",             "[OX2H]-c",                                   72.0, "UGT"),
    ("UGT_aliphatic_OH_primary", "[CX4;H2]-[OX2H]",                           74.5, "UGT"),
    ("UGT_aliphatic_OH_sec",     "[CX4;H1]-[OX2H]",                           75.8, "UGT"),
    ("UGT_aliphatic_OH_tert",    "[CX4;H0]-[OX2H]",                           77.2, "UGT"),
    ("UGT_primary_amine",        "[NX3;H2;!$(N-C=O)]-[CX4,c]",               76.3, "UGT"),
    ("UGT_secondary_amine",      "[NX3;H1;!$(N-C=O)]-[CX4,c]",               78.1, "UGT"),
    ("UGT_aromatic_amine",       "[NX3;H1,H2]-c",                              73.5, "UGT"),
    ("UGT_carboxylic_acid",      "[CX3](=O)[OX2H1]",                          80.4, "UGT"),
    ("UGT_acyl_glucuronide",     "[CX3](=O)-[OX2H0,OX1H0]",                  81.0, "UGT"),
    ("UGT_thiol",                "[SX2H]",                                     79.0, "UGT"),
    ("UGT_tetrazole_NH",         "[nH]1nnnn1",                                 74.0, "UGT"),

    # ══════════════════════════════════════════════════════════════════════
    # SULT (Sulfotransferase) — Phase II sulfation sites
    # Phenols > benzylic alcohols > primary amines
    # Reference: Gamage et al. Toxicol. Sci. 2006
    # ══════════════════════════════════════════════════════════════════════
    ("SULT_phenol",              "[OX2H]-c",                                   70.5, "SULT"),
    ("SULT_benzylic_OH",         "[CX4;H1,H2](-[OX2H])-c",                    73.2, "SULT"),
    ("SULT_primary_alc",         "[CX4;H2]-[OX2H]",                           76.0, "SULT"),
    ("SULT_aryl_amine",          "[NX3;H1,H2]-c",                              77.8, "SULT"),
    ("SULT_N_hydroxy",           "[NX3]-[OX2H]",                               71.0, "SULT"),
    ("SULT_hydroxamic_acid",     "[NX3;H1]-[CX3](=O)",                         78.5, "SULT"),

    # ══════════════════════════════════════════════════════════════════════
    # FMO (Flavin-containing Monooxygenase) — soft nucleophile oxidation
    # FMO1/FMO3 prefer tertiary amines and thioethers
    # Reference: Cashman, Curr. Drug Metab. 2000; Krueger & Williams 2005
    # ══════════════════════════════════════════════════════════════════════
    ("FMO_tertiary_amine",       "[NX3;H0;!$(N-C=O);!$(N~[!#6])]",           75.0, "FMO"),
    ("FMO_secondary_amine",      "[NX3;H1;!$(N-C=O);!$(N~[!#6])]",           77.5, "FMO"),
    ("FMO_thioether",            "[SX2;H0;!$(S=O);!$(S~[!#6])]",             68.0, "FMO"),
    ("FMO_thiol",                "[SX2H]",                                     65.0, "FMO"),
    ("FMO_disulfide",            "[SX2]-[SX2]",                               70.0, "FMO"),
    ("FMO_selenide",             "[SeX2H0]",                                   63.0, "FMO"),
    ("FMO_phosphine",            "[PX3;H0]",                                   66.5, "FMO"),
    ("FMO_hydrazine",            "[NX3;H1,H2]-[NX3;H1,H2]",                  72.0, "FMO"),
    ("FMO_N_oxide_precursor",    "[n;H0;+0]",                                  76.0, "FMO"),

    # ══════════════════════════════════════════════════════════════════════
    # AO (Aldehyde Oxidase) — azaheterocycle C-H oxidation
    # AO specifically attacks electron-poor azaheterocycle carbons
    # Reference: Pryde et al. J. Med. Chem. 2010; Lepri et al. 2017
    # ══════════════════════════════════════════════════════════════════════
    ("AO_pyrimidine_C",          "c1ncncn1",                                   78.0, "AO"),
    ("AO_pyridine_C2",           "c1ccncc1",                                   82.0, "AO"),
    ("AO_quinoxaline",           "c1cnc2ccccc2n1",                             76.5, "AO"),
    ("AO_phthalazine",           "c1ccc2cnncc2c1",                             74.0, "AO"),
    ("AO_purine_C8",             "c1nc2[nH]cnc2n1",                            77.0, "AO"),
    ("AO_imidazo_CH",            "c1ncc[nH]1",                                 80.0, "AO"),
    ("AO_azaheterocycle_CH",     "[cH]1[nH0][cH0,nH0][nH0,cH0][cH0,nH0]1",  83.0, "AO"),
    ("AO_aldehyde",              "[CX3H1](=O)",                                60.0, "AO"),

    # ══════════════════════════════════════════════════════════════════════
    # MAO (Monoamine Oxidase) — oxidative deamination of amines
    # MAO-A prefers serotonin/norepinephrine-type; MAO-B prefers benzylamine-type
    # Reference: Youdim et al. Nat Rev Neurosci 2006
    # ══════════════════════════════════════════════════════════════════════
    ("MAO_primary_amine_benzyl", "[NX3;H2]-[CX4;H2]-c",                       78.0, "MAO"),
    ("MAO_primary_amine_alkyl",  "[NX3;H2]-[CX4;H2]-[CX4]",                  80.5, "MAO"),
    ("MAO_secondary_amine",      "[NX3;H1;!$(N-C=O)]-[CX4;H2]-c",            79.2, "MAO"),
    ("MAO_indolyl_CH2_N",        "c1ccc2[nH]cc([CX4;H2]-[NX3])c2c1",         76.0, "MAO"),
    ("MAO_phenethylamine",       "c1ccccc1-[CX4;H2]-[CX4;H2]-[NX3;H2]",      77.5, "MAO"),

    # ══════════════════════════════════════════════════════════════════════
    # NAT (N-Acetyltransferase) — Phase II N-acetylation sites
    # Primary aromatic amines and sulfonamides
    # Reference: Hein et al. Carcinogenesis 2000
    # ══════════════════════════════════════════════════════════════════════
    ("NAT_aryl_primary_amine",   "[NX3;H2]-c",                                 73.0, "NAT"),
    ("NAT_sulfonamide_NH",       "[NX3;H1]-[SX4](=O)(=O)",                    75.5, "NAT"),
    ("NAT_hydrazide",            "[NX3;H1,H2]-[CX3](=O)-[NX3;H1,H2]",        77.0, "NAT"),

    # ══════════════════════════════════════════════════════════════════════
    # COMT (Catechol-O-Methyltransferase) — catechol O-methylation
    # Requires ortho-dihydroxybenzene (catechol) motif
    # Reference: Männistö & Kaakkola, Pharmacol. Rev. 1999
    # ══════════════════════════════════════════════════════════════════════
    ("COMT_catechol",            "c1cc([OH1])c([OH1])cc1",                     68.0, "COMT"),
    ("COMT_3_4_diOH",            "[OH1]c1ccc([OH1])cc1",                       69.5, "COMT"),

    # ══════════════════════════════════════════════════════════════════════
    # GST (Glutathione S-Transferase) — electrophilic soft spots
    # Michael acceptors and epoxide-like systems
    # Reference: Hayes et al. Annu. Rev. Pharmacol. 2005
    # ══════════════════════════════════════════════════════════════════════
    ("GST_michael_acceptor",     "[CX3;H0,H1]=[CX3]-[CX3](=O)",              74.0, "GST"),
    ("GST_alpha_beta_unsat",     "[CX3]=[CX3]-[C,c](=O)",                     72.5, "GST"),
    ("GST_halide_benzylic",      "[CX4;H1,H2](-[F,Cl,Br,I])-c",             71.0, "GST"),
    ("GST_epoxide",              "[CX4]1O[CX4]1",                             69.0, "GST"),
    ("GST_quinone",              "[#6]-1(=O)-[#6]=[#6]-[#6](=O)-[#6]=[#6]-1", 67.0, "GST"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetaboliteRecord:
    """
    Canonical container for a single predicted metabolite.

    All mass fields are in Da; they are set by MassSpecTracker after
    the record is created by a pipeline wrapper.
    """
    smiles_canonical: str                          # RDKit-canonicalised SMILES
    inchikey: str                                  # 27-char InChIKey for deduplication
    neutral_mass: float = 0.0                      # monoisotopic neutral mass (Da)
    delta_mass: str = ""                           # formatted delta vs. parent (+x.xxxx)
    adduct_mplus_h: float = 0.0                    # [M+H]+  theoretical m/z
    adduct_mminus_h: float = 0.0                   # [M-H]-  theoretical m/z
    source_pipeline: str = ""                      # "sygma" | "biotransformer" | "dl" | "smartcyp"
    reaction_label: str = ""                       # human-readable transformation name
    phase: int = 0                                 # 1 = Phase I, 2 = Phase II, 0 = unknown
    soft_spot_atoms: list[int] = field(default_factory=list)   # 0-based atom indices
    smartcyp_scores: list[dict[str, Any]] = field(default_factory=list)
    dl_confidence: float = 0.0                    # normalised [0, 1] from DL model
    ensemble_score: float = 0.0                   # final aggregated ranking score


# ---------------------------------------------------------------------------
# Mass Spec Tracker
# ---------------------------------------------------------------------------

class MassSpecTracker:
    """
    Computes monoisotopic masses and theoretical HRMS adducts for any
    parent/metabolite pair.

    Usage
    -----
    tracker = MassSpecTracker("COc1ccc(CC(C)N)cc1")   # amphetamine analogue
    tracker.annotate(record)                           # mutates record in-place
    """

    PROTON: float = PROTON_MASS

    def __init__(self, parent_smiles: str) -> None:
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        if parent_mol is None:
            raise ValueError(f"Cannot parse parent SMILES: {parent_smiles!r}")
        Chem.SanitizeMol(parent_mol)
        self.parent_smiles: str = Chem.MolToSmiles(parent_mol)
        self.parent_mass: float = round(
            Descriptors.ExactMolWt(parent_mol), MASS_PRECISION
        )
        log.debug("MassSpecTracker: parent=%s  mass=%.4f Da",
                  self.parent_smiles, self.parent_mass)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def neutral_mass(self, mol: Chem.Mol) -> float:
        """Return monoisotopic neutral mass rounded to MASS_PRECISION places."""
        return round(Descriptors.ExactMolWt(mol), MASS_PRECISION)

    def delta_mass_str(self, metabolite_mass: float) -> str:
        """
        Return delta mass vs. parent as a sign-prefixed string.

        Examples:  '+2.0157'   '-15.9949'   '+0.0000'
        """
        delta = round(metabolite_mass - self.parent_mass, MASS_PRECISION)
        sign = "+" if delta >= 0 else "-"
        return f"{sign}{abs(delta):.{MASS_PRECISION}f}"

    def adduct_mplus_h(self, neutral_mass: float) -> float:
        """[M+H]+ theoretical m/z = neutral_mass + proton_mass."""
        return round(neutral_mass + self.PROTON, MASS_PRECISION)

    def adduct_mminus_h(self, neutral_mass: float) -> float:
        """[M-H]- theoretical m/z = neutral_mass − proton_mass."""
        return round(neutral_mass - self.PROTON, MASS_PRECISION)

    def annotate(self, record: MetaboliteRecord) -> None:
        """
        Mutate *record* in-place, filling all four mass fields.
        Requires record.smiles_canonical to already be set.
        """
        mol = Chem.MolFromSmiles(record.smiles_canonical)
        if mol is None:
            log.warning("MassSpecTracker.annotate: invalid SMILES %r — skipping",
                        record.smiles_canonical)
            return
        nm = self.neutral_mass(mol)
        record.neutral_mass   = nm
        record.delta_mass     = self.delta_mass_str(nm)
        record.adduct_mplus_h  = self.adduct_mplus_h(nm)
        record.adduct_mminus_h = self.adduct_mminus_h(nm)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _mol_to_record(mol: Chem.Mol,
                   source: str,
                   label: str = "",
                   phase: int = 0) -> MetaboliteRecord | None:
    """
    Convert an RDKit Mol to a MetaboliteRecord, computing canonical SMILES
    and InChIKey.  Returns None if the mol is invalid.
    """
    try:
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        if not smi:
            return None
        # MolToInchi is unavailable on some Windows RDKit builds; fall back to
        # a SHA-256 of canonical SMILES as a stable cross-engine dedup key.
        inchikey = ""
        try:
            inchi = Chem.MolToInchi(mol)  # type: ignore[attr-defined]
            if inchi:
                inchikey = rdMolDescriptors.CalcInchiKey(inchi)
        except Exception:
            pass
        if not inchikey:
            import hashlib
            inchikey = "SMIKEY-" + hashlib.sha256(smi.encode()).hexdigest()[:20]
        return MetaboliteRecord(
            smiles_canonical=smi,
            inchikey=inchikey,
            source_pipeline=source,
            reaction_label=label,
            phase=phase,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("_mol_to_record: skipped (%s): %s", exc, Chem.MolToSmiles(mol))
        return None


def _smiles_to_record(smiles: str,
                      source: str,
                      label: str = "",
                      phase: int = 0) -> MetaboliteRecord | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _mol_to_record(mol, source=source, label=label, phase=phase)


# ---------------------------------------------------------------------------
# Pipeline A — SyGMa Phase I / II rule-based tree
# ---------------------------------------------------------------------------

class SyGMaPipeline:
    """
    Wraps the 3D-e-Chem/sygma library to generate Phase I and Phase II
    metabolites via sequential SMARTS reaction transforms.

Dual-Track Execution Engine
    ---------------------------
    Track 1 (Sequential): Phase I on the parent, then Phase II on each Phase I
        product -- oxidations and sequential Phase I+II conjugates (+192 Da).
    Track 2 (Direct Phase II Bypass): Phase II applied DIRECTLY to the raw parent
        -- direct parent O-/N-glucuronidation and sulfation (+176.0321 Da) that
        Track 1 structurally cannot reach.

    The two trees are merged and deduplicated on RDKit canonical SMILES
    (Chem.CanonSmiles), and direct parent glucuronides are tagged explicitly.

    Parameters
    ----------
    phase1_cycles : int
        Number of successive Phase I rule applications (default 1).
    phase2_cycles : int
        Number of successive Phase II rule applications (default 1).
    """

    def __init__(self, phase1_cycles: int = 1, phase2_cycles: int = 1) -> None:
        self.phase1_cycles = phase1_cycles
        self.phase2_cycles = phase2_cycles

    @staticmethod
    def _rule_path(phase: str) -> str:
        """Return absolute path to SyGMa rule file for the given phase name."""
        import os
        rules_dir = os.path.join(os.path.dirname(sygma.__file__), "rules")
        return os.path.join(rules_dir, f"{phase}.txt")

    # Exact mass of a glucuronic-acid conjugation (C6H8O6) — the signature of a
    # direct parent O-/N-glucuronide produced by the Track 2 bypass.
    _GLUCURONIDE_DELTA: float = 176.0321

    # ------------------------------------------------------------------
    # Track 1 — Sequential Pathway (Phase I -> Phase II)
    # ------------------------------------------------------------------
    def _track1_sequential(self, parent_mol):
        """Phase I (phase1_cycles) on the parent, then Phase II (phase2_cycles) on
        each Phase I product. Captures oxidations and sequential Phase I+II
        conjugates (e.g. hydroxylation + glucuronidation, +192 Da).
        Returns a list of (metabolite_mol, sygma_pathway_label, phase) tuples."""
        out = []
        parent_smi = Chem.MolToSmiles(parent_mol)

        scenario_p1 = sygma.Scenario([[sygma.ruleset["phase1"], self.phase1_cycles]])
        tree_p1 = scenario_p1.run(parent_mol)
        tree_p1.calc_scores()

        phase1_mols = []
        for entry in tree_p1.to_list():
            mol = entry.get("SyGMa_metabolite")
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                continue
            # Skip the parent passthrough: applying Phase II to the unchanged parent
            # is Track 2's job, so excluding it keeps direct-parent conjugates unique
            # to Track 2 and correctly tagged.
            if smi == parent_smi:
                continue
            label = (entry.get("SyGMa_pathway") or "").strip().rstrip(";").strip()
            out.append((mol, label, 1))
            phase1_mols.append(mol)

        for p1_mol in phase1_mols:
            try:
                scenario_p2 = sygma.Scenario([[sygma.ruleset["phase2"], self.phase2_cycles]])
                tree_p2 = scenario_p2.run(p1_mol)
                tree_p2.calc_scores()
            except Exception as exc:
                log.debug("Track 1: Phase II on a Phase I product failed: %s", exc)
                continue
            for entry in tree_p2.to_list():
                mol = entry.get("SyGMa_metabolite")
                if mol is None:
                    continue
                label = (entry.get("SyGMa_pathway") or "").strip().rstrip(";").strip()
                out.append((mol, label, 2))
        return out

    # ------------------------------------------------------------------
    # Track 2 — Direct Phase II Bypass (Phase II applied to the raw parent)
    # ------------------------------------------------------------------
    def _track2_direct_phase2(self, parent_mol):
        """Run a Phase II scenario DIRECTLY on the raw parent, bypassing Phase I:
            direct_phase2_scenario = sygma.Scenario([[sygma.ruleset['phase2'], 1]])
        This forces SyGMa to enumerate direct parent O-glucuronidation,
        N-glucuronidation and sulfation that the sequential Track 1 cannot reach.
        Returns a list of (metabolite_mol, sygma_pathway_label, phase) tuples."""
        out = []
        direct_phase2_scenario = sygma.Scenario([[sygma.ruleset["phase2"], 1]])
        tree = direct_phase2_scenario.run(parent_mol)
        tree.calc_scores()
        for entry in tree.to_list():
            mol = entry.get("SyGMa_metabolite")
            if mol is None:
                continue
            label = (entry.get("SyGMa_pathway") or "").strip().rstrip(";").strip()
            out.append((mol, label, 2))
        return out

    # ------------------------------------------------------------------
    # Dual-Track Execution Engine
    # ------------------------------------------------------------------
    def run(self, parent_smiles: str) -> list[MetaboliteRecord]:
        """Dual-Track Execution Engine.

        Track 1 (Sequential): Phase I -> Phase II. Oxidations + sequential
            Phase I+II conjugates (e.g. ox + glucuronide, +192 Da).
        Track 2 (Direct Phase II Bypass): Phase II applied directly to the parent.
            Direct parent O-/N-glucuronidation and sulfation (+176.0321 Da) that
            Track 1 structurally cannot produce.

        The two trees are merged and deduplicated using RDKit canonical SMILES
        (Chem.CanonSmiles) as dictionary keys, so no structure is double-counted.
        Track 1 is ingested first (sequential provenance wins ties); direct parent
        glucuronides are unique to Track 2 and are tagged explicitly."""
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        if parent_mol is None:
            raise ValueError(f"SyGMaPipeline: invalid SMILES {parent_smiles!r}")
        parent_smi = Chem.MolToSmiles(parent_mol)
        try:
            parent_mass = Descriptors.ExactMolWt(parent_mol)
        except Exception:
            parent_mass = 0.0

        # Run both tracks independently (one failing must not lose the other)
        try:
            track1 = self._track1_sequential(parent_mol)
        except Exception as exc:
            log.warning("SyGMa Track 1 (sequential Phase I->II) failed: %s", exc)
            track1 = []
        try:
            track2 = self._track2_direct_phase2(parent_mol)
        except Exception as exc:
            log.warning("SyGMa Track 2 (direct Phase II bypass) failed: %s", exc)
            track2 = []

        # Merge + deduplicate on canonical SMILES
        merged = {}

        def _ingest(items, track):
            for mol, pathway, phase in items:
                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    continue
                if not smi or smi == parent_smi:
                    continue
                try:
                    key = Chem.CanonSmiles(smi)
                except Exception:
                    key = smi
                if key in merged:
                    continue  # first-writer-wins (Track 1 ingested first)
                label = pathway
                if track == 2:
                    try:
                        delta = Descriptors.ExactMolWt(mol) - parent_mass
                    except Exception:
                        delta = 0.0
                    if abs(delta - self._GLUCURONIDE_DELTA) < 0.01:
                        label = "Direct Parent Phase II Glucuronidation (+176.0321 Da)"
                    else:
                        label = f"Direct Parent Phase II: {pathway or 'conjugation'}"
                rec = _mol_to_record(mol, source="sygma", label=label, phase=phase)
                if rec:
                    merged[key] = rec

        _ingest(track1, 1)
        _ingest(track2, 2)

        records = list(merged.values())
        n_direct = sum(1 for r in records
                       if r.reaction_label.startswith("Direct Parent Phase II"))
        log.info(
            "SyGMaPipeline (dual-track): %d unique metabolites after dedup "
            "(Track1 raw=%d, Track2 raw=%d; %d direct-parent Phase II conjugates)",
            len(records), len(track1), len(track2), n_direct,
        )
        return records


# ---------------------------------------------------------------------------
# Pipeline B — BioTransformer JAR subprocess wrapper
# ---------------------------------------------------------------------------

class BioTransformerPipeline:
    """
    Drives the BioTransformer 3.0 JAR via a subprocess call, parsing the
    output SDF to recover predicted metabolites.

    Parameters
    ----------
    jar_path : str | Path
        Filesystem path to BioTransformer3.0.jar (resolved from env var
        BIOTRANSFORMER_JAR or the data/ directory by default).
    transformation_type : str
        BioTransformer transformation type string.
        Common values: "allHuman", "cyp450", "ecbased", "gutmicro",
        "hgut", "phase1", "phase2", "superbio".
    n_steps : int
        Number of metabolic steps (default 2).
    java_heap_mb : int
        JVM -Xmx heap in MB (default 512).
    """

    _DEFAULT_JAR_SEARCH_PATHS: list[Path] = [
        Path("data/biotransformer/BioTransformer3.0.jar"),
        Path("BioTransformer3.0.jar"),
    ]

    def __init__(
        self,
        jar_path: str | Path | None = None,
        transformation_type: str = "allHuman",
        n_steps: int = 2,
        java_heap_mb: int = 512,
    ) -> None:
        self.jar_path = self._resolve_jar(jar_path)
        self.transformation_type = transformation_type
        self.n_steps = n_steps
        self.java_heap_mb = java_heap_mb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_jar(self, candidate: str | Path | None) -> Path:
        """Find the JAR, honouring an env var override first."""
        env_jar = os.environ.get("BIOTRANSFORMER_JAR")
        if env_jar:
            p = Path(env_jar)
            if p.is_file():
                return p
            log.warning("BIOTRANSFORMER_JAR env var set but file not found: %s", p)

        if candidate is not None:
            p = Path(candidate)
            if p.is_file():
                return p
            raise FileNotFoundError(f"BioTransformer JAR not found at {p}")

        for path in self._DEFAULT_JAR_SEARCH_PATHS:
            if path.is_file():
                return path

        raise FileNotFoundError(
            "BioTransformer JAR not found.  Set BIOTRANSFORMER_JAR env var "
            "or place BioTransformer3.0.jar in data/biotransformer/."
        )

    def _write_smiles_input(self, smiles: str, tmp_dir: str) -> Path:
        """Write a single-SMILES input file understood by BioTransformer -ismi."""
        input_path = Path(tmp_dir) / "input.smi"
        input_path.write_text(smiles.strip() + "\n", encoding="utf-8")
        return input_path

    def _parse_output_sdf(self, sdf_path: Path) -> list[MetaboliteRecord]:
        """
        Read the BioTransformer output SDF.  Each molecule's SD properties
        include 'Reaction' and 'Biosystem', which are captured as the label.
        """
        records: list[MetaboliteRecord] = []
        if not sdf_path.is_file() or sdf_path.stat().st_size == 0:
            log.warning("BioTransformerPipeline: output SDF empty or missing: %s",
                        sdf_path)
            return records

        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
        for mol in supplier:
            if mol is None:
                continue
            reaction = mol.GetPropsAsDict().get("Reaction", "")
            biosystem = mol.GetPropsAsDict().get("Biosystem", "")
            label = f"{biosystem}:{reaction}" if biosystem else reaction
            rec = _mol_to_record(mol, source="biotransformer", label=label, phase=0)
            if rec:
                records.append(rec)

        log.info("BioTransformerPipeline: parsed %d records from SDF", len(records))
        return records

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, parent_smiles: str) -> list[MetaboliteRecord]:
        """
        Execute BioTransformer for *parent_smiles* and return metabolites.

        The subprocess call pattern is:
            java -Xmx{heap}m -jar BioTransformer3.0.jar \
                 -ismi input.smi \
                 -oformat sdf \
                 -osdf output.sdf \
                 -btType {type} \
                 -nsteps {n}
        """
        with tempfile.TemporaryDirectory(prefix="hrms_bt_") as tmp_dir:
            input_path  = self._write_smiles_input(parent_smiles, tmp_dir)
            output_path = Path(tmp_dir) / "output.sdf"

            cmd: list[str] = [
                "java",
                f"-Xmx{self.java_heap_mb}m",
                "-jar", str(self.jar_path),
                "-ismi",    str(input_path),
                "-oformat", "sdf",
                "-osdf",    str(output_path),
                "-btType",  self.transformation_type,
                "-nsteps",  str(self.n_steps),
            ]

            log.debug("BioTransformerPipeline: cmd=%s", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,   # 5-minute hard ceiling
                )
            except subprocess.TimeoutExpired:
                log.error("BioTransformerPipeline: JAR timed out after 300 s")
                return []
            except FileNotFoundError:
                log.error("BioTransformerPipeline: 'java' not found on PATH")
                return []

            if result.returncode != 0:
                log.error("BioTransformerPipeline: JAR exited %d\nSTDERR:\n%s",
                           result.returncode, result.stderr[:2000])
                return []

            return self._parse_output_sdf(output_path)


# ---------------------------------------------------------------------------
# Pipeline C — DL Transformer inference (MetaTrans / Meta-Predictor)
# ---------------------------------------------------------------------------

class MetaboliteTransformerPredictor:
    """
    Object-oriented inference handler that unifies the MetaTrans and
    Meta-Predictor paradigms under a single predict() interface.

    Architecture overview
    ---------------------
    • Both models are seq2seq Transformers that accept tokenised SMILES
      as input and decode predicted metabolite SMILES (MetaTrans) or
      per-atom site-of-metabolism probability vectors (Meta-Predictor).
    • Model weights are loaded from ``model_dir`` via
      ``transformers.AutoModelForSeq2SeqLM`` (MetaTrans) and
      ``transformers.AutoModelForTokenClassification`` (Meta-Predictor).
    • Inference runs on CPU by default; pass device="cuda:0" if a GPU is
      available and the environment.yml torch pin includes a CUDA build.

    Parameters
    ----------
    model_dir : str | Path
        Root directory for model checkpoints.  Expected sub-dirs:
            models/metatrans/      — seq2seq weights
            models/metapredictor/  — token-classification weights
    device : str
        PyTorch device string ("cpu", "cuda:0", etc.).
    max_new_tokens : int
        Token budget for MetaTrans beam-search decoding.
    num_beams : int
        Beam width for MetaTrans decoding.
    batch_size : int
        Number of SMILES strings processed per forward pass.
    """

    _SMILES_CHAR_VOCAB: list[str] = list(
        "BCFHINOPSbcnosp"
        "0123456789"
        "()[]=#@\\/%+-."
        "ClBrSi"   # digraph tokens handled by splitting in tokenise()
    )

    def __init__(
        self,
        model_dir: str | Path = "models",
        device: str = "cpu",
        max_new_tokens: int = 128,
        num_beams: int = 5,
        batch_size: int = 8,
    ) -> None:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.batch_size = batch_size

        model_dir = Path(model_dir)

        # ── MetaTrans: SMILES → SMILES seq2seq ──
        metatrans_path = model_dir / "metatrans"
        log.info("Loading MetaTrans from %s …", metatrans_path)
        self.metatrans_tokenizer = AutoTokenizer.from_pretrained(
            str(metatrans_path), use_fast=True
        )
        self.metatrans_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(metatrans_path)
        ).to(self.device)
        self.metatrans_model.eval()

        # ── Meta-Predictor: per-atom SoM token classification ──
        metapred_path = model_dir / "metapredictor"
        log.info("Loading Meta-Predictor from %s …", metapred_path)
        self.metapred_tokenizer = AutoTokenizer.from_pretrained(
            str(metapred_path), use_fast=True
        )
        self.metapred_model = AutoModelForTokenClassification.from_pretrained(
            str(metapred_path)
        ).to(self.device)
        self.metapred_model.eval()

        log.info(
            "MetaboliteTransformerPredictor ready on %s  "
            "(metatrans params=%s  metapred params=%s)",
            device,
            sum(p.numel() for p in self.metatrans_model.parameters()),
            sum(p.numel() for p in self.metapred_model.parameters()),
        )

    # ------------------------------------------------------------------
    # SMILES tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def tokenise_smiles(smiles: str) -> list[str]:
        """
        Character-level SMILES tokeniser that correctly handles two-letter
        element symbols (Cl, Br, Si) and bracket groups as single tokens.

        Returns a flat list of string tokens, e.g.:
            "ClC(=O)N"  →  ["Cl", "C", "(", "=", "O", ")", "N"]
        """
        tokens: list[str] = []
        i = 0
        two_char_atoms = {"Cl", "Br", "Si", "Se", "Na", "Mg", "Al",
                          "Ca", "Cu", "Zn", "Fe", "As"}
        while i < len(smiles):
            # bracket atom: [NH], [13C], [OH-], etc.
            if smiles[i] == "[":
                j = smiles.index("]", i) + 1
                tokens.append(smiles[i:j])
                i = j
            # two-character element symbols
            elif i + 1 < len(smiles) and smiles[i : i + 2] in two_char_atoms:
                tokens.append(smiles[i : i + 2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
        return tokens

    @staticmethod
    def _batch(seq: list, size: int):
        """Yield successive fixed-size chunks from *seq*."""
        for start in range(0, len(seq), size):
            yield seq[start : start + size]

    # ------------------------------------------------------------------
    # MetaTrans: SMILES → SMILES seq2seq prediction
    # ------------------------------------------------------------------

    def predict_structures(self, smiles_list: list[str]) -> list[list[str]]:
        """
        Run MetaTrans beam-search decoding on each SMILES in *smiles_list*.

        Returns a list (one entry per input) of decoded candidate SMILES
        strings (up to ``num_beams`` per input after deduplication).

        The tokeniser receives space-separated character tokens:
            "COc1ccc" → "C O c 1 c c c"
        which matches MetaTrans's published training-data format.
        """
        results: list[list[str]] = []

        for batch in self._batch(smiles_list, self.batch_size):
            # Space-tokenise each SMILES for MetaTrans
            token_strings = [" ".join(self.tokenise_smiles(s)) for s in batch]

            enc = self.metatrans_tokenizer(
                token_strings,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.metatrans_model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    early_stopping=True,
                )

            # Decode: outputs shape = (batch_size * num_beams, seq_len)
            decoded_all = self.metatrans_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Re-group: num_beams sequences per input SMILES
            for i in range(len(batch)):
                seqs = decoded_all[i * self.num_beams : (i + 1) * self.num_beams]
                # Strip whitespace between tokens, deduplicate, validate with RDKit
                candidates: list[str] = []
                seen: set[str] = set()
                for seq in seqs:
                    raw = seq.replace(" ", "")
                    mol = Chem.MolFromSmiles(raw)
                    if mol is not None:
                        canonical = Chem.MolToSmiles(mol)
                        if canonical not in seen:
                            seen.add(canonical)
                            candidates.append(canonical)
                results.append(candidates)

        return results

    # ------------------------------------------------------------------
    # Meta-Predictor: per-atom SoM probability vector
    # ------------------------------------------------------------------

    def predict_som_probabilities(
        self, smiles_list: list[str]
    ) -> list[dict[str, Any]]:
        """
        Run Meta-Predictor token-classification on *smiles_list*.

        Returns one dict per input SMILES:
        {
            "smiles":      str,
            "tokens":      list[str],          # tokenised input
            "probs":       list[float],        # per-token SoM probability [0, 1]
            "top_atoms":   list[int],          # 0-based token indices, descending prob
        }

        The model assigns a label (0 = non-SoM, 1 = SoM) to every input
        token; the softmax probability of label-1 is used as the score.
        """
        results: list[dict[str, Any]] = []

        for batch in self._batch(smiles_list, self.batch_size):
            tokenised = [self.tokenise_smiles(s) for s in batch]
            # Encode as space-separated token strings
            token_strs = [" ".join(t) for t in tokenised]

            enc = self.metapred_tokenizer(
                token_strs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                is_split_into_words=False,
            ).to(self.device)

            with torch.no_grad():
                logits = self.metapred_model(**enc).logits  # (B, seq, num_labels)

            probs_batch = torch.softmax(logits, dim=-1)     # (B, seq, num_labels)

            for i, smi in enumerate(batch):
                n_tokens = len(tokenised[i])
                # Slice to the actual (non-padded) token count
                # Label index 1 = SoM positive class
                som_probs: list[float] = probs_batch[i, :n_tokens, 1].tolist()
                top_atoms = sorted(
                    range(len(som_probs)),
                    key=lambda k: som_probs[k],
                    reverse=True,
                )
                results.append(
                    {
                        "smiles":    smi,
                        "tokens":    tokenised[i],
                        "probs":     [round(p, 4) for p in som_probs],
                        "top_atoms": top_atoms,
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Unified predict → MetaboliteRecord
    # ------------------------------------------------------------------

    def run(self, parent_smiles: str) -> list[MetaboliteRecord]:
        """
        Full DL prediction pass:
          1. MetaTrans → candidate SMILES → MetaboliteRecords
          2. Meta-Predictor SoM probabilities → soft_spot_atoms annotation
             on the parent record (confidence attached to each candidate).
        """
        candidates = self.predict_structures([parent_smiles])[0]
        som_result = self.predict_som_probabilities([parent_smiles])[0]

        records: list[MetaboliteRecord] = []
        for smi in candidates:
            rec = _smiles_to_record(smi, source="dl",
                                    label="MetaTrans_seq2seq", phase=1)
            if rec is None:
                continue
            # Attach the global SoM top-atom list from Meta-Predictor
            # as a structural annotation on each DL-generated metabolite
            rec.soft_spot_atoms = som_result["top_atoms"][: 10]

            # Confidence: mean probability of the top-5 predicted SoM tokens
            top5_probs = [
                som_result["probs"][k]
                for k in som_result["top_atoms"][:5]
                if k < len(som_result["probs"])
            ]
            rec.dl_confidence = round(
                sum(top5_probs) / len(top5_probs) if top5_probs else 0.0, 4
            )
            records.append(rec)

        log.info("MetaboliteTransformerPredictor: %d DL metabolites", len(records))
        return records


# ---------------------------------------------------------------------------
# Pipeline D — SMARTCyp P450 soft-spot profiler
# ---------------------------------------------------------------------------

class SMARTCypProfiler:
    """
    Fragment-based SMARTS lookup profiler implementing the SMARTCyp 2.4.2
    rule set for CYP3A4, CYP2D6, and CYP2C9 site-of-metabolism prediction.

    For each matching SMARTS pattern the DFT-derived activation energy
    (Ea, kcal/mol) is recorded.  Lower Ea = higher metabolic liability.
    Reactive atoms are extracted directly from the substructure match
    indices and stored on each MetaboliteRecord.

    Parameters
    ----------
    ea_cutoff : float
        Only patterns with Ea ≤ ea_cutoff are considered (default 95.0
        kcal/mol covers the full published SMARTCyp range).
    """

    def __init__(self, ea_cutoff: float = 95.0) -> None:
        self.ea_cutoff = ea_cutoff
        # Pre-compile all SMARTS patterns for efficiency
        self._compiled: list[tuple[str, Chem.Mol, float, str]] = []
        for label, smarts, ea, isoform in SMARTCYP_RULES:
            if ea > ea_cutoff:
                continue
            pattern_mol = Chem.MolFromSmarts(smarts)
            if pattern_mol is None:
                log.warning("SMARTCypProfiler: cannot parse SMARTS %r — skipped",
                            smarts)
                continue
            self._compiled.append((label, pattern_mol, ea, isoform))
        log.debug("SMARTCypProfiler: %d compiled rules (Ea ≤ %.1f)",
                  len(self._compiled), ea_cutoff)

    # ------------------------------------------------------------------

    def profile(self, smiles: str) -> dict[str, Any]:
        """
        Run all SMARTCyp rules against *smiles*.

        Returns
        -------
        {
          "smiles":         str,
          "matches":        [
              {
                "rule":     str,        # pattern label
                "isoform":  str,        # CYP isoform
                "ea":       float,      # DFT Ea (kcal/mol)
                "atoms":    list[int],  # 0-based heavy-atom indices (first match)
              }, ...
          ],
          "soft_spot_atoms": list[int], # union of atoms across all matching rules
          "top_rule":        str | None # label of the lowest-Ea rule
        }
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning("SMARTCypProfiler.profile: invalid SMILES %r", smiles)
            return {"smiles": smiles, "matches": [],
                    "soft_spot_atoms": [], "top_rule": None}

        matches_out: list[dict[str, Any]] = []
        all_atoms: set[int] = set()

        for label, pattern, ea, isoform in self._compiled:
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                continue
            # Use the first match tuple's first atom as the primary reactive site
            first_match_atoms = list(matches[0])
            matches_out.append(
                {
                    "rule":    label,
                    "isoform": isoform,
                    "ea":      ea,
                    "atoms":   first_match_atoms,
                }
            )
            all_atoms.update(first_match_atoms)

        # Sort by ascending Ea (most reactive first)
        matches_out.sort(key=lambda m: m["ea"])
        top_rule = matches_out[0]["rule"] if matches_out else None

        return {
            "smiles":          smiles,
            "matches":         matches_out,
            "soft_spot_atoms": sorted(all_atoms),
            "top_rule":        top_rule,
        }

    def run(self, parent_smiles: str) -> list[MetaboliteRecord]:
        """
        Return a single MetaboliteRecord for the parent itself annotated
        with SMARTCyp scores (soft_spot_atoms and smartcyp_scores fields).

        The profiler does not generate new metabolite structures; it
        annotates the metabolic vulnerabilities of the input molecule.
        A "pseudo-record" for the parent is returned so that
        aggregate_metabolism() can include the soft-spot annotation in
        the output matrix alongside the structure-generating pipelines.
        """
        profile = self.profile(parent_smiles)

        mol = Chem.MolFromSmiles(parent_smiles)
        if mol is None:
            return []

        rec = _mol_to_record(mol, source="smartcyp",
                             label="SMARTCyp_parent_profile", phase=0)
        if rec is None:
            return []

        rec.soft_spot_atoms   = profile["soft_spot_atoms"]
        rec.smartcyp_scores   = profile["matches"]
        return [rec]


# ---------------------------------------------------------------------------
# Deduplication + ensemble scoring
# ---------------------------------------------------------------------------

def _deduplicate(records: list[MetaboliteRecord]) -> list[MetaboliteRecord]:
    """
    Collapse records with identical InChIKey, merging source pipelines and
    accumulating soft_spot_atoms.  The highest dl_confidence is kept.
    """
    seen: dict[str, MetaboliteRecord] = {}
    for rec in records:
        if rec.inchikey not in seen:
            seen[rec.inchikey] = rec
        else:
            existing = seen[rec.inchikey]
            # Merge pipeline attribution
            if rec.source_pipeline not in existing.source_pipeline:
                existing.source_pipeline += f"|{rec.source_pipeline}"
            # Union soft-spot atoms
            existing.soft_spot_atoms = sorted(
                set(existing.soft_spot_atoms) | set(rec.soft_spot_atoms)
            )
            # Keep highest DL confidence
            if rec.dl_confidence > existing.dl_confidence:
                existing.dl_confidence = rec.dl_confidence
            # Accumulate SMARTCyp scores
            existing.smartcyp_scores.extend(rec.smartcyp_scores)
    return list(seen.values())


def _score_ensemble(
    records: list[MetaboliteRecord],
    weights: dict[str, float] | None = None,
) -> list[MetaboliteRecord]:
    """
    Assign a final ensemble_score to each record.

    Scoring components
    ------------------
    • Source coverage bonus:  +0.15 for each pipeline that predicted this
      metabolite (max 4 × 0.15 = 0.60).
    • DL confidence:          dl_confidence × weight["dl"]  (default 0.25).
    • SMARTCyp reactivity:    normalised inverse-Ea score × weight["sc"]
      (default 0.15).  Lower Ea → higher score.
    • Phase 1 bonus:          +0.05 if phase == 1 (generally more reliable).

    All scores are clamped to [0.0, 1.0].
    """
    _w = {"dl": 0.25, "sc": 0.15}
    if weights:
        _w.update(weights)

    pipeline_names = {"sygma", "biotransformer", "dl", "smartcyp"}

    # Determine Ea normalisation range across all records
    all_ea_values = [
        m["ea"]
        for rec in records
        for m in rec.smartcyp_scores
    ]
    ea_min = min(all_ea_values, default=80.0)
    ea_max = max(all_ea_values, default=95.0)
    ea_range = ea_max - ea_min if ea_max > ea_min else 1.0

    for rec in records:
        score = 0.0

        # Source coverage
        sources = set(rec.source_pipeline.split("|"))
        coverage = len(sources & pipeline_names)
        score += coverage * 0.15

        # DL confidence
        score += rec.dl_confidence * _w["dl"]

        # SMARTCyp inverse-Ea contribution
        if rec.smartcyp_scores:
            best_ea = min(m["ea"] for m in rec.smartcyp_scores)
            normalised_ea = (ea_max - best_ea) / ea_range   # high score = low Ea
            score += normalised_ea * _w["sc"]

        # Phase 1 bonus
        if rec.phase == 1:
            score += 0.05

        rec.ensemble_score = round(min(max(score, 0.0), 1.0), 4)

    records.sort(key=lambda r: r.ensemble_score, reverse=True)
    return records


# ---------------------------------------------------------------------------
# Local scaffold-specific fragmentation layer (custom, editable)
# ---------------------------------------------------------------------------
# Our chemical series (chromanone / isoindolinone / anthranilate scaffold)
# undergoes recurring, specific cleavages and conjugations that the generic
# SyGMa rule set does not model.  This layer injects local reaction SMARTS that
# are applied DIRECTLY to the parent after the rule engines run.
#
# Each entry is: (name, reaction_smarts, target_delta)
#   * name             human-readable pathway label
#   * reaction_smarts  an AllChem.ReactionFromSmarts transform.  For a cleavage,
#                      split a bond into H-capped fragments ("...>>A.B"); the
#                      engine keeps the largest sanitised product fragment.
#   * target_delta     experimentally observed Da shift the rule is meant to
#                      reproduce (annotation / QC only).
#
# EDIT ME: the exact -121 / -123 / -156 / -105 losses are chemotype-specific and
# are NOT simple single-bond cleavages of the parent (verified empirically); to
# reproduce them precisely, paste the bond-cleavage SMARTS taken from the
# experimental structure proposals into the list below.
SCAFFOLD_FRAGMENTATION: list[tuple[str, str, float]] = [
    # -- Methylation / alkylation additions (+14.0157 Da) -- validated exact --
    ("O-Methylation (carboxyl -> methyl ester)",
     "[CX3:1](=[OX1:2])[OX2H]>>[CX3:1](=[OX1:2])[OX2]C", 14.0157),
    ("N-Methylation (aryl secondary amine)",
     "[c:1][NX3;H1:2][#6:3]>>[c:1][NX3:2]([#6:3])C", 14.0157),
    ("O-Methylation (phenol)",
     "[c:1][OX2H:2]>>[c:1][OX2:2]C", 14.0157),
    # -- Scaffold cleavages (TEMPLATES -- refine SMARTS to hit the exact losses) --
    ("Anthranilate arm C-N cleavage",
     "[c:1][NX3:2][CX4:3]>>[c:1][NX3:2].[CX4:3]", -121.0),
    ("Lactam N-dealkylation",
     "[CX3:1](=[OX1:2])[NX3:3][CX4:4]>>[CX3:1](=[OX1:2])[NX3:3].[CX4:4]", -156.0),
]

_COMPILED_FRAG: list[tuple[str, Any, float, str]] = []
for _frag_name, _frag_sm, _frag_td in SCAFFOLD_FRAGMENTATION:
    try:
        _frag_rxn = AllChem.ReactionFromSmarts(_frag_sm)
    except Exception:
        _frag_rxn = None
    _COMPILED_FRAG.append((_frag_name, _frag_rxn, _frag_td, _frag_sm))


def generate_scaffold_fragments(parent_smiles: str,
                                min_heavy_atoms: int = 5) -> list[MetaboliteRecord]:
    """
    Apply the local SCAFFOLD_FRAGMENTATION reaction SMARTS directly to the parent.

    For each rule the parent is matched; if the transform fires, open valencies
    are cleaned (sanitisation), the largest product fragment is retained, and its
    exact monoisotopic mass and signed delta vs parent are recorded.  Cleavage
    products are tagged 'Scaffold-Specific Fragment Cleavage'; additions
    (e.g. +14.0157 methylation) are tagged 'Scaffold-Specific Methylation/Addition'.

    Returns a list of MetaboliteRecord, deduplicated on canonical SMILES.
    """
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        return []
    parent_mass = Descriptors.ExactMolWt(parent_mol)
    parent_smi = Chem.MolToSmiles(parent_mol)

    out: dict[str, MetaboliteRecord] = {}
    for name, rxn, target_delta, _sm in _COMPILED_FRAG:
        if rxn is None:
            continue
        try:
            product_sets = rxn.RunReactants((parent_mol,))
        except Exception:
            continue
        for product_set in product_sets:
            best = None
            for prod in product_set:
                try:
                    Chem.SanitizeMol(prod)
                except Exception:
                    continue
                for piece in Chem.GetMolFrags(prod, asMols=True, sanitizeFrags=False):
                    try:
                        Chem.SanitizeMol(piece)
                    except Exception:
                        continue
                    if piece.GetNumHeavyAtoms() < min_heavy_atoms:
                        continue
                    if best is None or Descriptors.ExactMolWt(piece) > Descriptors.ExactMolWt(best):
                        best = piece
            if best is None:
                continue
            try:
                smi = Chem.MolToSmiles(best)
            except Exception:
                continue
            if not smi or smi == parent_smi:
                continue
            try:
                key = Chem.CanonSmiles(smi)
            except Exception:
                key = smi
            if key in out:
                continue
            delta = Descriptors.ExactMolWt(best) - parent_mass
            # Gate: only emit when the rule reproduces its declared target shift
            # (additions to 0.05 Da; cleavages to 2 Da). This keeps the layer
            # precise — unrefined cleavage templates stay silent until correct
            # SMARTS are supplied.
            tol = 0.05 if target_delta >= 0 else 2.0
            if abs(delta - target_delta) > tol:
                continue
            if target_delta >= 0:
                label = f"Scaffold-Specific Methylation/Addition ({name}, {delta:+.4f} Da)"
            else:
                label = f"Scaffold-Specific Fragment Cleavage ({name}, {delta:+.4f} Da)"
            rec = _mol_to_record(best, source="scaffold_frag", label=label, phase=1)
            if rec:
                out[key] = rec
    return list(out.values())



# ---------------------------------------------------------------------------
# DMPK-empirical prioritisation & regioisomer capping
# ---------------------------------------------------------------------------
# Replaces the flat 0.200 rule-engine baseline with a confidence score tuned to
# mass-spec relevance, and caps redundant regioisomers so the screening list is
# not swamped by theoretical ring-hydroxylation positions that are never seen.

_PRIORITY_HIGH = 0.70
_PRIORITY_MED = 0.40


def _prio_delta(met: dict) -> float:
    try:
        return float(met.get("delta_mass", 0))
    except (TypeError, ValueError):
        return 0.0


def _prio_family(met: dict) -> str:
    """Transformation family used for grouping/scoring. Prefers the human-readable
    transformation_type (set by the API layer); falls back to the raw label."""
    s = ((met.get("transformation_type") or "") + " " +
         (met.get("reaction_label") or "")).lower()
    if "aromatic hydroxylation" in s:        return "aromatic_hydroxylation"
    if "aliphatic hydroxylation" in s:       return "aliphatic_hydroxylation"
    if "benzylic hydroxylation" in s:        return "benzylic_hydroxylation"
    if "scaffold-specific fragment" in s:    return "scaffold_cleavage"
    if "glucuron" in s:                      return "glucuronidation"
    if "sulfat" in s or "sulfon" in s:       return "sulfation"
    if "methylation" in s and "dealk" not in s and "demethyl" not in s:
        return "methylation"
    if "dealkylation" in s or "demethyl" in s: return "dealkylation"
    if "dehydrogen" in s or "desatur" in s:  return "dehydrogenation"
    return (met.get("transformation_type") or met.get("reaction_label") or "other"
            ).lower().split("(")[0].strip()[:24] or "other"


def _empirical_confidence(met: dict) -> float:
    """DMPK-empirical confidence (0-1) by mass-spec relevance."""
    fam = _prio_family(met)
    delta = _prio_delta(met)
    if fam == "glucuronidation":
        if abs(delta - 176.0321) < 0.05:   return 0.85   # direct parent glucuronide
        if abs(delta - 192.0270) < 0.05:   return 0.75   # oxidative glucuronide
        return 0.70
    if fam == "scaffold_cleavage":         return 0.80   # -121 / -156 ... cleavages
    if fam == "sulfation":                 return 0.72
    if fam in ("aliphatic_hydroxylation", "benzylic_hydroxylation"):
        return 0.60                                       # accessible, unhindered
    if fam == "dealkylation":              return 0.55
    if fam == "methylation":               return 0.55
    if fam == "aromatic_hydroxylation":    return 0.15   # crowded / hindered ring
    if fam == "dehydrogenation":           return 0.40
    return 0.30


def _prio_accessibility(met: dict) -> float:
    """Proxy for structural accessibility / model attention, used to rank
    regioisomers within a group: higher DL confidence and lower SMARTCyp
    activation energy (more reactive site) rank first."""
    dl = met.get("dl_confidence", 0.0) or 0.0
    sc = met.get("smartcyp_scores") or []
    ea_score = 0.0
    if sc:
        best_ea = min((s.get("ea", 999) for s in sc), default=999)
        ea_score = max(0.0, (100.0 - best_ea) / 100.0)
    return 0.5 * float(dl) + ea_score


def prioritize_predictions(metabolite_list: list[dict],
                           max_isomers_per_group: int = 3,
                           max_conjugates_per_class: int = 3) -> list[dict]:
    """
    DMPK-empirical prioritisation and regioisomer capping.

    1. Cap regioisomers: metabolites sharing a transformation family AND the same
       nominal mass shift (e.g. 8 aromatic ring-hydroxylation positions, all
       +15.9949 Da) are grouped; only the top `max_isomers_per_group` by
       accessibility / model attention survive, discarding redundant position
       isomers that clutter the list.
    2. Dynamic empirical weighting: each survivor is rescored 0-1 by mass-spec
       relevance (direct +176 glucuronide -> 0.85; scaffold cleavage -> 0.80;
       accessible aliphatic oxidation -> 0.60; hindered aromatic hydroxylation
       -> 0.15; ...), replacing the flat 0.200 baseline.
    3. Sort strictly by the new confidence score, descending.

    Returns a new ordered, capped list; input dicts are not mutated.
    """
    from collections import defaultdict
    work = [dict(m) for m in metabolite_list]          # copy — do not mutate input

    # Conjugate classes are collapsed across mass-variants so theoretical
    # glucuronide/sulfate isomers (and SyGMa +/-2H mass artefacts) do not flood
    # the screening list; oxidations stay grouped by exact mass so each
    # position-isomer family is capped independently.
    conj_classes = {"glucuronidation", "sulfation"}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for met in work:
        fam = _prio_family(met)
        key = (fam,) if fam in conj_classes else (fam, round(_prio_delta(met)))
        groups[key].append(met)

    kept: list[dict] = []
    for key, members in groups.items():
        members.sort(key=_prio_accessibility, reverse=True)
        if len(key) == 1:  # conjugate class — keep distinct masses, capped
            seen_mass: set[int] = set()
            picked: list[dict] = []
            for m in members:
                nd = round(_prio_delta(m))
                if nd in seen_mass:
                    continue
                seen_mass.add(nd)
                picked.append(m)
                if len(picked) >= max_conjugates_per_class:
                    break
            kept.extend(picked)
        else:
            kept.extend(members[:max_isomers_per_group])

    for met in kept:
        score = _empirical_confidence(met)
        met["ensemble_score"] = round(score, 3)
        met["confidence_label"] = (
            "High Confidence" if score >= _PRIORITY_HIGH
            else "Medium Confidence" if score >= _PRIORITY_MED
            else "Low Confidence"
        )

    kept.sort(key=lambda m: m.get("ensemble_score", 0.0), reverse=True)
    return kept



# ---------------------------------------------------------------------------
# Public aggregate entry point
# ---------------------------------------------------------------------------

def aggregate_metabolism(
    parent_smiles: str,
    *,
    run_sygma: bool = True,
    run_biotransformer: bool = True,
    run_dl: bool = True,
    run_smartcyp: bool = True,
    biotransformer_jar: str | Path | None = None,
    biotransformer_type: str = "allHuman",
    dl_model_dir: str | Path = "models",
    dl_device: str = "cpu",
    sygma_phase1_cycles: int = 1,
    sygma_phase2_cycles: int = 1,
    smartcyp_ea_cutoff: float = 95.0,
    run_scaffold_frag: bool = True,
    ensemble_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Master orchestrator: run all enabled pipelines, deduplicate, score,
    and return the output matrix.

    Parameters
    ----------
    parent_smiles : str
        Input molecule as a SMILES string.
    run_sygma / run_biotransformer / run_dl / run_smartcyp : bool
        Toggle individual pipeline execution.
    biotransformer_jar : str | Path | None
        Path override for BioTransformer3.0.jar.
    biotransformer_type : str
        BioTransformer transformation type (default "allHuman").
    dl_model_dir : str | Path
        Root directory for DL model checkpoints.
    dl_device : str
        PyTorch device string for DL inference.
    sygma_phase1_cycles / sygma_phase2_cycles : int
        SyGMa rule application cycles per phase.
    smartcyp_ea_cutoff : float
        Maximum Ea (kcal/mol) threshold for SMARTCyp rules.
    ensemble_weights : dict | None
        Custom weight overrides for ensemble scoring.

    Returns
    -------
    {
        "parent": {
            "smiles":             str,
            "neutral_mass":       float,    # monoisotopic, Da
            "adduct_mplus_h":     float,    # [M+H]+
            "adduct_mminus_h":    float,    # [M-H]-
        },
        "metabolites": [
            {
                "smiles_canonical":  str,
                "inchikey":          str,
                "neutral_mass":      float,
                "delta_mass":        str,   # e.g. "+15.9949"
                "adduct_mplus_h":    float,
                "adduct_mminus_h":   float,
                "source_pipeline":   str,   # pipe-delimited if multi-source
                "reaction_label":    str,
                "phase":             int,
                "soft_spot_atoms":   list[int],
                "smartcyp_scores":   list[dict],
                "dl_confidence":     float,
                "ensemble_score":    float,
            },
            ...
        ],
        "soft_spot_summary": {
            "top_atoms":   list[int],   # union across all metabolites
            "top_rules":   list[str],   # most frequent SMARTCyp rule labels
        },
        "pipeline_stats": {
            "sygma_count":          int,
            "biotransformer_count": int,
            "dl_count":             int,
            "smartcyp_count":       int,
            "total_after_dedup":    int,
        },
    }
    """
    # ── Validate & canonicalise parent ──
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        raise ValueError(f"aggregate_metabolism: invalid parent SMILES {parent_smiles!r}")
    Chem.SanitizeMol(parent_mol)
    canonical_parent = Chem.MolToSmiles(parent_mol)

    tracker = MassSpecTracker(canonical_parent)

    all_records: list[MetaboliteRecord] = []
    stats: dict[str, int] = {
        "sygma_count": 0,
        "biotransformer_count": 0,
        "dl_count": 0,
        "smartcyp_count": 0,
    }

    # ── Pipeline A: SyGMa ──
    if run_sygma:
        try:
            sygma_pipe = SyGMaPipeline(
                phase1_cycles=sygma_phase1_cycles,
                phase2_cycles=sygma_phase2_cycles,
            )
            sygma_records = sygma_pipe.run(canonical_parent)
            stats["sygma_count"] = len(sygma_records)
            all_records.extend(sygma_records)
        except Exception:
            log.exception("Pipeline A (SyGMa) failed — continuing")

    # ── Pipeline B: BioTransformer ──
    if run_biotransformer:
        try:
            bt_pipe = BioTransformerPipeline(
                jar_path=biotransformer_jar,
                transformation_type=biotransformer_type,
            )
            bt_records = bt_pipe.run(canonical_parent)
            stats["biotransformer_count"] = len(bt_records)
            all_records.extend(bt_records)
        except FileNotFoundError as e:
            log.warning("Pipeline B (BioTransformer) skipped: %s", e)
        except Exception:
            log.exception("Pipeline B (BioTransformer) failed — continuing")

    # ── Pipeline C: DL Transformer ──
    if run_dl:
        try:
            dl_pipe = MetaboliteTransformerPredictor(
                model_dir=dl_model_dir,
                device=dl_device,
            )
            dl_records = dl_pipe.run(canonical_parent)
            stats["dl_count"] = len(dl_records)
            all_records.extend(dl_records)
        except Exception:
            log.exception("Pipeline C (DL Transformer) failed — continuing")

    # ── Pipeline D: SMARTCyp ──
    if run_smartcyp:
        try:
            sc_pipe = SMARTCypProfiler(ea_cutoff=smartcyp_ea_cutoff)
            sc_records = sc_pipe.run(canonical_parent)
            stats["smartcyp_count"] = len(sc_records)
            all_records.extend(sc_records)
        except Exception:
            log.exception("Pipeline D (SMARTCyp) failed — continuing")

    # ── Custom scaffold-specific fragmentation layer (Pipeline E) ──
    if run_scaffold_frag:
        try:
            frag_records = generate_scaffold_fragments(canonical_parent)
            stats["scaffold_frag_count"] = len(frag_records)
            all_records.extend(frag_records)
        except Exception:
            log.exception("Scaffold fragmentation layer failed — continuing")

    # ── Annotate all records with mass spec data ──
    for rec in all_records:
        tracker.annotate(rec)

    # ── Deduplicate on InChIKey ──
    deduped = _deduplicate(all_records)

    # ── Ensemble scoring ──
    scored = _score_ensemble(deduped, weights=ensemble_weights)
    stats["total_after_dedup"] = len(scored)

    # ── Soft-spot summary ──
    all_soft_atoms: set[int] = set()
    rule_counts: dict[str, int] = {}
    atom_best: dict[int, tuple[float, str]] = {}  # atom -> (score, isoform)
    for rec in scored:
        all_soft_atoms.update(rec.soft_spot_atoms)
        for match in rec.smartcyp_scores:
            rule = match.get("rule", "")
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
            try:
                ea = float(match.get("ea", 99.0))
            except (TypeError, ValueError):
                ea = 99.0
            score = max(0.0, min(1.0, (95.0 - ea) / 25.0))
            iso = match.get("isoform", "")
            for a in match.get("atoms", []):
                if a not in atom_best or score > atom_best[a][0]:
                    atom_best[a] = (score, iso)

    top_rules = sorted(rule_counts, key=lambda r: rule_counts[r], reverse=True)[:5]
    # Per-atom soft-spot scores, computed from the full set BEFORE any display
    # capping/prioritisation, so the soft-spot map is independent of which
    # metabolites are shown in the ranked table.
    soft_spot_atom_scores = [
        {"atom_idx": a, "score": round(sc, 4), "isoform": iso}
        for a, (sc, iso) in sorted(atom_best.items())
    ]

    # ── Build output matrix ──
    output: dict[str, Any] = {
        "parent": {
            "smiles":          canonical_parent,
            "neutral_mass":    tracker.parent_mass,
            "adduct_mplus_h":  tracker.adduct_mplus_h(tracker.parent_mass),
            "adduct_mminus_h": tracker.adduct_mminus_h(tracker.parent_mass),
        },
        "metabolites": [
            {
                "smiles_canonical":  r.smiles_canonical,
                "inchikey":          r.inchikey,
                "neutral_mass":      r.neutral_mass,
                "delta_mass":        r.delta_mass,
                "adduct_mplus_h":    r.adduct_mplus_h,
                "adduct_mminus_h":   r.adduct_mminus_h,
                "source_pipeline":   r.source_pipeline,
                "reaction_label":    r.reaction_label,
                "phase":             r.phase,
                "soft_spot_atoms":   r.soft_spot_atoms,
                "smartcyp_scores":   r.smartcyp_scores,
                "dl_confidence":     r.dl_confidence,
                "ensemble_score":    r.ensemble_score,
            }
            for r in scored
        ],
        "soft_spot_summary": {
            "top_atoms": sorted(all_soft_atoms),
            "top_rules": top_rules,
            "atom_scores": soft_spot_atom_scores,
        },
        "pipeline_stats": stats,
    }

    log.info(
        "aggregate_metabolism complete: %d unique metabolites "
        "(sygma=%d  bt=%d  dl=%d  sc=%d)",
        stats["total_after_dedup"],
        stats["sygma_count"],
        stats["biotransformer_count"],
        stats["dl_count"],
        stats["smartcyp_count"],
    )
    return output
