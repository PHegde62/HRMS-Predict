"""
app/main.py
===========
HRMS Predictor — FastAPI Web Server

Endpoints
---------
POST /predict
    Validate SMILES → run all four pipelines → apply ensemble consensus
    filter → return structured JSON payload with full HRMS data matrix.

POST /render-soft-spots
    Accept SMILES + per-atom risk scores → draw 2D molecule with a
    translucent risk-proportional colour overlay → return SVG.

GET  /health
    Liveness probe for container orchestrators / CI checks.

CORS is configured to allow the local Streamlit frontend (port 8501)
to communicate with this backend (port 8000) without browser blocking.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ── Pydantic v2 (bundled with FastAPI ≥ 0.111) ───────────────────────────────
from pydantic import BaseModel, Field, field_validator

# ── RDKit ────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# ── Our engine ───────────────────────────────────────────────────────────────
# Adjust sys.path so this module resolves whether run from the project root
# (`uvicorn app.main:app`) or from inside `app/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.engine.metabolism import aggregate_metabolism  # noqa: E402
from app.engine.metabolism import prioritize_predictions  # noqa: E402
from app.engine.metabolism_improvements import apply_all_improvements  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hrms.api")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HRMS Metabolite Predictor",
    description=(
        "Aggregates SyGMa, BioTransformer, MetaTrans, Meta-Predictor and "
        "SMARTCyp into a single HRMS-annotated metabolite prediction service."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the local Streamlit UI (port 8501) to reach this server
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    # Explicit origins rather than ["*"] so credentials / cookies work if
    # authentication is added later.
    allow_origins=[
        "http://localhost:8501",    # Streamlit default
        "http://127.0.0.1:8501",
        "http://0.0.0.0:8501",
        # Extend this list for any production deployment origin.
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

# ── /predict ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Body accepted by POST /predict.

    All pipeline toggles default to True so a minimal request body of
    ``{"smiles": "CCO"}`` works without specifying options.
    """
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Parent molecule as a SMILES string.",
        examples=["COc1ccc(CC(C)N)cc1"],
    )
    run_sygma: bool = Field(True,  description="Enable SyGMa Phase I/II pipeline.")
    run_biotransformer: bool = Field(True,  description="Enable BioTransformer pipeline.")
    run_dl: bool = Field(True,  description="Enable DL transformer pipeline.")
    run_smartcyp: bool = Field(True,  description="Enable SMARTCyp soft-spot profiler.")
    biotransformer_jar: str | None = Field(
        None,
        description="Explicit path to BioTransformer3.0.jar (overrides env var).",
    )
    biotransformer_type: str = Field(
        "allHuman",
        description="BioTransformer transformation type.",
    )
    dl_model_dir: str = Field(
        "models",
        description="Root directory for DL model checkpoints.",
    )
    dl_device: str = Field(
        "cpu",
        description="PyTorch device string ('cpu', 'cuda:0', …).",
    )
    sygma_phase1_cycles: int = Field(1, ge=1, le=3)
    sygma_phase2_cycles: int = Field(1, ge=1, le=3)
    smartcyp_ea_cutoff: float = Field(
        95.0,
        ge=70.0,
        le=110.0,
        description="Maximum DFT activation energy (kcal/mol) for SMARTCyp rules.",
    )
    # ── Output control ────────────────────────────────────────────────────────
    top_n: int = Field(
        12,
        ge=1,
        le=500,
        description=(
            "Number of metabolites to display in the ranked section of the report. "
            "Any integer from 1 to 500. Default 12 matches the original PDF layout."
        ),
    )
    return_all: bool = Field(
        False,
        description=(
            "If True, the response includes ALL predicted metabolites (not just top_n) "
            "in a separate `all_metabolites` list, suitable for a full LC-HRMS target list. "
            "The ranked `metabolites` list still contains only top_n entries."
        ),
    )
    # ── Improvement toggles ───────────────────────────────────────────────────
    run_sequential: bool = Field(
        False,
        description="Generate Phase I→II sequential metabolites via SyGMa (~2× slower).",
    )
    run_ndealk: bool = Field(
        True,
        description="Add targeted N-dealkylation products (piperazine N-aryl, N-methyl).",
    )
    run_reduction: bool = Field(
        True,
        description="Add carbonyl/N-oxide reduction products (+2H, −O).",
    )

    prioritize: bool = Field(
        True,
        description=(
            "Apply DMPK-empirical prioritisation: cap redundant regioisomers "
            "and rescore by mass-spec relevance (direct glucuronide / scaffold "
            "cleavage = High; hindered aromatic hydroxylation = Low), then sort "
            "by the new confidence score."
        ),
    )

    @field_validator("smiles")
    @classmethod
    def smiles_must_be_parseable(cls, v: str) -> str:
        """Reject SMILES that RDKit cannot parse before touching any pipeline."""
        mol = Chem.MolFromSmiles(v.strip())
        if mol is None:
            raise ValueError(
                f"RDKit cannot parse the supplied SMILES string: {v!r}.  "
                "Please check for typos or unsupported notation."
            )
        return v.strip()


class HRMSAdducts(BaseModel):
    mplus_h: float  = Field(..., description="[M+H]+ theoretical m/z (Da)")
    mminus_h: float = Field(..., description="[M-H]- theoretical m/z (Da)")


class ParentMetrics(BaseModel):
    smiles: str
    neutral_mass: float = Field(..., description="Monoisotopic neutral mass (Da)")
    molecular_formula: str
    adducts: HRMSAdducts


class SMARTCypMatch(BaseModel):
    rule: str
    isoform: str
    ea: float = Field(..., description="DFT activation energy (kcal/mol)")
    atoms: list[int]


class MetaboliteEntry(BaseModel):
    rank: int
    smiles_canonical: str
    inchikey: str
    molecular_formula: str
    neutral_mass: float = Field(..., description="Monoisotopic neutral mass (Da)")
    delta_mass: str     = Field(..., description="Δ mass vs. parent (signed, 4 d.p.)")
    adducts: HRMSAdducts
    source_pipeline: str = Field(..., description="Pipe-delimited pipeline names")
    reaction_label: str
    phase: int          = Field(..., description="1=Phase I, 2=Phase II, 0=unknown")
    soft_spot_atoms: list[int]
    smartcyp_scores: list[SMARTCypMatch]
    dl_confidence: float = Field(..., ge=0.0, le=1.0)
    ensemble_score: float = Field(..., ge=0.0, le=1.0)
    # ── Consensus fields (added by the API layer) ──────────────────────────
    consensus_verified: bool = Field(
        False,
        description=(
            "True when this metabolite was independently predicted by BOTH "
            "a rule-based path (SyGMa or BioTransformer) AND a DL model "
            "(MetaTrans/Meta-Predictor)."
        ),
    )
    confidence_label: str = Field(
        "",
        description=(
            "'High Confidence (Consensus Verified)' for dual-path agreement, "
            "'Rule-Based Only', 'DL Only', or 'Single Source' otherwise."
        ),
    )
    # ── Transformation annotation (added by the API layer) ─────────────────
    transformation_type: str = Field(
        "",
        description="Human-readable transformation class, e.g. 'Aromatic Hydroxylation'.",
    )
    responsible_enzyme: str = Field(
        "",
        description="Primary enzyme class responsible, e.g. 'CYP2C9', 'UGT', 'FMO'.",
    )


class SoftSpotSummary(BaseModel):
    top_atoms: list[int]
    top_rules: list[str]
    atom_scores: list[dict[str, Any]] = Field(default_factory=list)


class PipelineStats(BaseModel):
    sygma_count: int
    biotransformer_count: int
    dl_count: int
    smartcyp_count: int
    total_after_dedup: int
    consensus_count: int = 0
    elapsed_seconds: float = 0.0


class PredictResponse(BaseModel):
    parent: ParentMetrics
    metabolites: list[MetaboliteEntry]
    all_metabolites: list[MetaboliteEntry] = Field(
        default_factory=list,
        description=(
            "Full list of all predicted metabolites (populated only when "
            "return_all=True in the request). Same schema as `metabolites`."
        ),
    )
    soft_spot_summary: SoftSpotSummary
    pipeline_stats: PipelineStats


# ── /render-soft-spots ───────────────────────────────────────────────────────

class AtomRiskScore(BaseModel):
    """Per-atom risk entry supplied by the client."""
    atom_idx: int = Field(..., ge=0, description="0-based heavy-atom index")
    score: float  = Field(..., ge=0.0, le=1.0, description="Normalised risk [0, 1]")
    isoform: str  = Field("", description="Optional isoform label for tooltip")


class SoftSpotRenderRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=4096)
    atom_scores: list[AtomRiskScore] = Field(
        ...,
        description=(
            "List of atom-index / risk-score pairs.  Atoms not in this list "
            "are rendered without highlight."
        ),
    )
    width: int  = Field(600, ge=200, le=1600, description="SVG canvas width (px)")
    height: int = Field(400, ge=200, le=1200, description="SVG canvas height (px)")
    highlight_alpha_max: float = Field(
        0.70,
        ge=0.1,
        le=1.0,
        description="Opacity ceiling for the highest-risk atom highlight.",
    )
    colour_scheme: str = Field(
        "risk",
        description=(
            "'risk' = red→orange gradient by score; "
            "'isoform' = fixed colour per CYP isoform."
        ),
    )

    @field_validator("smiles")
    @classmethod
    def smiles_parseable(cls, v: str) -> str:
        if Chem.MolFromSmiles(v.strip()) is None:
            raise ValueError(f"Unparseable SMILES: {v!r}")
        return v.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Isoform → (R, G, B) base colour used when colour_scheme == "isoform"
# CYP isoforms: warm reds/oranges
# Phase II conjugation enzymes: cool greens/blues
# Oxidoreductases (FMO, AO, MAO): purples/violets
_ISOFORM_RGB: dict[str, tuple[float, float, float]] = {
    # ── CYP isoforms ──
    "CYP3A4": (0.95, 0.30, 0.18),   # coral-red
    "CYP2D6": (0.20, 0.50, 0.90),   # steel-blue
    "CYP2C9": (0.95, 0.60, 0.10),   # amber-orange
    "CYP1A2": (0.85, 0.20, 0.50),   # crimson-pink
    "CYP2C19":(0.30, 0.70, 0.80),   # cyan-teal
    "CYP2E1": (0.75, 0.25, 0.85),   # violet
    # ── Phase II conjugation enzymes ──
    "UGT":    (0.10, 0.75, 0.45),   # emerald-green
    "SULT":   (0.15, 0.55, 0.85),   # royal-blue
    "GST":    (0.20, 0.80, 0.60),   # mint-green
    "NAT":    (0.05, 0.65, 0.55),   # dark-teal
    "COMT":   (0.85, 0.45, 0.10),   # burnt-orange
    # ── Non-CYP oxidoreductases ──
    "FMO":    (0.90, 0.75, 0.05),   # golden-yellow
    "AO":     (0.65, 0.20, 0.80),   # purple
    "MAO":    (0.80, 0.30, 0.70),   # magenta-violet
}
_ISOFORM_RGB_DEFAULT = (0.60, 0.60, 0.65)  # neutral grey for unknown


def _molecular_formula(smiles: str) -> str:
    """Return Hill-order molecular formula from a valid SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    # Add implicit Hs for formula computation, then strip them from the mol
    mol_with_h = Chem.AddHs(mol)
    return rdMolDescriptors.CalcMolFormula(mol_with_h)


# ---------------------------------------------------------------------------
# Transformation type and enzyme classifier
# ---------------------------------------------------------------------------
#
# The −14.0157 Da ambiguity problem
# ----------------------------------
# A Δm/z of −14.0157 can arise from two completely different reactions:
#   A) CYP-mediated N- or O-demethylation  (removes −CH2 from N-CH3 or O-CH3)
#   B) COMT/TPMT-mediated de-methylation   (reversal of O- or N-methylation)
#
# SyGMa's reaction label is the primary discriminator:
#   "N-dealkylation_(N-CH3)"  → always CYP
#   "O-dealkylation_(O-CH3)"  → always CYP
#   "methylation_(O-methylation)" or "methylation_(N-methylation)" → COMT/TPMT
#
# If the label is ambiguous or absent, we apply SMARTS-based structural
# context on the PARENT molecule:
#   - Parent has N-CH3 adjacent to aromatic ring  → CYP demethylation
#   - Parent has O-CH3 on aromatic ring (methoxy) → CYP demethylation
#   - Parent has catechol OH → COMT methylation is plausible
#   - Otherwise → CYP demethylation (more common in drugs)
# ---------------------------------------------------------------------------

# SMARTS patterns for structural context disambiguation
_SMARTS_N_METHYL_ARYL   = Chem.MolFromSmarts("[NX3;!$(N-C=O)]-[CH3]")       # CYP N-demethylation
_SMARTS_O_METHYL_ARYL   = Chem.MolFromSmarts("[OX2]-[CH3]")                  # CYP O-demethylation
_SMARTS_CATECHOL        = Chem.MolFromSmarts("c1cc([OH1])c([OH1])cc1")        # COMT substrate
_SMARTS_N_METHYL_ALKYL  = Chem.MolFromSmarts("[NX3;H0;!$(N-C=O)]-[CH3]")    # tertiary N-methyl

# Map SyGMa pathway label keywords → (transformation_type, enzyme_class)
# SyGMa uses SMARTS reaction strings as pathway names (e.g. "[n:1][CH3]>>[nH1:1]")
# so we match BOTH readable text fragments AND SMARTS fragments.
#
# Matching priority order:
#   1. SMARTS reactant pattern fragments (>>, [n:, [N:, [O:, etc.)
#   2. Human-readable substrings (for BioTransformer and future engines)
#
# The −14 Da ambiguity (CYP demethylation vs COMT methylation) is resolved by
# detecting whether the SMARTS removes a CH3 from N or O (CYP demethylation)
# vs adds an O-methyl or N-methyl (COMT) — the reaction arrow direction ">>"
# and the reactant pattern make this unambiguous in SMARTS notation.

# SMARTS fragment patterns for classifying SyGMa reaction strings
# Each tuple: (smarts_fragment, transformation_type, enzyme)
# Checked BEFORE the text keyword map — SyGMa will always hit these.
_SMARTS_REACTION_MAP: list[tuple[str, str, str]] = [
    # ── N-demethylation patterns (reactant has N-CH3, product loses CH3) ──────
    # "[n:1][CH3]>>[nH1:1]"  — aromatic N-methyl removal (caffeine, theophylline)
    ("[ch3]>>[nh",          "N-Dealkylation",           "CYP1A2"),
    ("[n:1][ch3]",          "N-Dealkylation",           "CYP1A2 / CYP3A4"),
    # "[NH0;X3][CH3]" — tertiary amine N-demethylation
    ("[nh0;x3:2]([ch3]",    "N-Dealkylation",           "CYP3A4 / CYP2D6"),
    ("[nh1;x3:2][ch3]",     "N-Dealkylation",           "CYP2D6 / CYP3A4"),
    # Generic N-CH3 removal (any N-methyl)
    ("][ch3]>>[n",          "N-Dealkylation",           "CYP"),
    ("[n:2][ch3]",          "N-Dealkylation",           "CYP"),
    # ── O-demethylation patterns ──────────────────────────────────────────────
    # "[#6:1][O:2][CH3]>>[*:1][O:2]" — aryl/alkyl methoxy removal
    ("[o:2][ch3]>>[",       "O-Dealkylation",           "CYP2D6"),
    ("[#6:1][o:2][ch3]",    "O-Dealkylation",           "CYP2D6 / CYP3A4"),
    ("[c:1][o:2][ch3]",     "O-Dealkylation",           "CYP2D6"),
    # ── Aromatic hydroxylation ────────────────────────────────────────────────
    ("[ch1:5][a:6][a:7]1>>[",   "Aromatic Hydroxylation",   "CYP"),
    ("[ch1;$(",                 "Aromatic Hydroxylation",   "CYP"),
    ("[ch1:4]",                 "Aromatic Hydroxylation",   "CYP"),
    # ── Aliphatic hydroxylation ───────────────────────────────────────────────
    ("[ch2:1][ch2",          "Aliphatic Hydroxylation",  "CYP"),
    ("[cx4:1][ch2",          "Aliphatic Hydroxylation",  "CYP"),
    ("[c:1][ch1:2]",         "Benzylic Hydroxylation",   "CYP"),
    # ── N-oxidation ───────────────────────────────────────────────────────────
    ("[n;x3:2]([c",          "N-Oxidation",              "CYP / FMO"),
    ("[n+:2]([c",            "N-Oxidation",              "CYP / FMO"),
    # ── S-oxidation ───────────────────────────────────────────────────────────
    ("[s;x2:2][c",           "S-Oxidation",              "CYP / FMO"),
    ("[s;x3:2](=[o",         "S-Oxidation",              "CYP"),
    # ── Carbonyl / ketone / aldehyde oxidation ────────────────────────────────
    ("[ch1:2]=[o:3]",        "Oxidation to Carboxylic Acid", "AO / CYP"),
    # ── Reduction ────────────────────────────────────────────────────────────
    ("=[o:3])[c",            "Reduction (Carbonyl)",     "CYP / Reductase"),
    # ── Hydrolysis ────────────────────────────────────────────────────────────
    ("c(=o)o[c",             "Ester Hydrolysis",         "Esterase"),
    ("c(=o)[nh",             "Amide Hydrolysis",         "Amidase"),
    ("[nx2:1]=[ch1:2]",      "Imine Hydrolysis",         "Amidase"),
    # ── Ring closure ─────────────────────────────────────────────────────────
    ("[oh1][c:2]!@",         "Ring Closure / Cyclisation","Spontaneous"),
    ("[nh1;!$(nc=o):1][#6:2]~!@", "Ring Closure / Cyclisation","Spontaneous"),
    # ── Desaturation ─────────────────────────────────────────────────────────
    ("[cx4@!h0",             "Desaturation",             "CYP"),
    # ── Epoxidation ──────────────────────────────────────────────────────────
    ("[c:1]1o[c:2]1",        "Epoxidation",              "CYP"),
]

# Human-readable keyword map — used for BioTransformer labels and fallback
# NOTE: dealkylation patterns listed BEFORE methylation
_TRANSFORMATION_MAP: list[tuple[str, str, str]] = [
    ("aromatic_hydroxylation",  "Aromatic Hydroxylation",    "CYP"),
    ("aliphatic_hydroxylation", "Aliphatic Hydroxylation",   "CYP"),
    ("benzylic_hydroxylation",  "Benzylic Hydroxylation",    "CYP"),
    ("allylic_hydroxylation",   "Allylic Hydroxylation",     "CYP"),
    ("n-oxidation",             "N-Oxidation",               "CYP / FMO"),
    ("s-oxidation",             "S-Oxidation",               "CYP / FMO"),
    ("epoxidation",             "Epoxidation",               "CYP"),
    ("desaturation",            "Desaturation",              "CYP"),
    ("dehydrogenation",         "Dehydrogenation",           "CYP"),
    ("n-dealkylation",          "N-Dealkylation",            "CYP"),
    ("o-dealkylation",          "O-Dealkylation",            "CYP"),
    ("s-dealkylation",          "S-Dealkylation",            "CYP"),
    ("deamination",             "Oxidative Deamination",     "MAO"),
    ("reduction",               "Reduction",                 "CYP / Reductase"),
    ("ring_opening",            "Ring Opening",              "CYP"),
    ("ring_closure",            "Ring Closure / Cyclisation","Spontaneous"),
    ("hydrolysis",              "Hydrolysis",                "Esterase / Amidase"),
    ("ester_hydrolysis",        "Ester Hydrolysis",          "Esterase"),
    ("amide_hydrolysis",        "Amide Hydrolysis",          "Amidase"),
    ("glucuronidation",         "Glucuronidation",           "UGT"),
    ("glucuroni",               "Glucuronidation",           "UGT"),
    ("sulfation",               "Sulfation",                 "SULT"),
    ("sulfoconjugation",        "Sulfation",                 "SULT"),
    ("glutathione",             "Glutathione Conjugation",   "GST"),
    ("gsh_conjugation",         "Glutathione Conjugation",   "GST"),
    ("acetylation",             "N-Acetylation",             "NAT"),
    # methylation is LAST — only reached when SyGMa explicitly names it
    ("methylation",             "Methylation",               "COMT / TPMT"),
    ("glycine_conjugation",     "Glycine Conjugation",       "Glycine N-acyltransferase"),
    ("taurine_conjugation",     "Taurine Conjugation",       "Bile acid CoA transferase"),
    ("3a4_",                    "CYP3A4-mediated Oxidation", "CYP3A4"),
    ("2d6_",                    "CYP2D6-mediated Oxidation", "CYP2D6"),
    ("2c9_",                    "CYP2C9-mediated Oxidation", "CYP2C9"),
    ("ugt_",                    "Glucuronidation",           "UGT"),
    ("sult_",                   "Sulfation",                 "SULT"),
    ("fmo_",                    "FMO-mediated Oxidation",    "FMO"),
    ("ao_",                     "Aldehyde Oxidase Oxidation","AO"),
    ("mao_",                    "Oxidative Deamination",     "MAO"),
    ("nat_",                    "N-Acetylation",             "NAT"),
    ("comt_",                   "O-Methylation",             "COMT"),
    ("gst_",                    "Glutathione Conjugation",   "GST"),
]

# Enzyme isoform → canonical enzyme class name
_ISOFORM_TO_CLASS: dict[str, str] = {
    "CYP3A4": "CYP3A4", "CYP2D6": "CYP2D6", "CYP2C9": "CYP2C9",
    "CYP1A2": "CYP1A2", "CYP2C19":"CYP2C19", "CYP2E1": "CYP2E1",
    "UGT":  "UGT",  "SULT": "SULT",  "FMO":  "FMO",
    "AO":   "AO",   "MAO":  "MAO",   "NAT":  "NAT",
    "COMT": "COMT", "GST":  "GST",
}

# SMARTS patterns for structural context — enzyme isoform refinement
_DEMETHYLATION_ISOFORM_SMARTS: list[tuple[str, str]] = [
    ("[NX3;H0;!$(N-C=O)]-[CH3]",  "CYP2D6 / CYP3A4"),
    ("[NX3;H1;!$(N-C=O)]-[CH3]",  "CYP3A4"),
    ("c-[OX2]-[CH3]",             "CYP2D6"),
    ("[OX2]-[CH3]-[!#1]",         "CYP2D6 / CYP2C9"),
]
_COMPILED_DEMETH: list[tuple[object, str]] = [
    (Chem.MolFromSmarts(sm), iso)
    for sm, iso in _DEMETHYLATION_ISOFORM_SMARTS
    if Chem.MolFromSmarts(sm) is not None
]


def _structural_demethylation_enzyme(parent_smiles: str) -> str:
    if not parent_smiles:
        return "CYP (predicted)"
    try:
        mol = Chem.MolFromSmiles(parent_smiles)
        if mol is None:
            return "CYP (predicted)"
        for pattern, isoform in _COMPILED_DEMETH:
            if mol.HasSubstructMatch(pattern):
                return isoform
    except Exception:
        pass
    return "CYP (predicted)"


def _is_true_methylation(label_lower: str) -> bool:
    """True only for explicit Phase II methylation, not CYP demethylation."""
    return "methylation" in label_lower and "dealkylation" not in label_lower


def _classify_transformation(
    reaction_label: str,
    smartcyp_scores: list[dict],
    source_pipeline: str,
    phase: int,
    parent_smiles: str = "",
) -> tuple[str, str]:
    """
    Return (transformation_type, responsible_enzyme).

    SyGMa reaction labels are raw SMARTS strings (e.g. "[n:1][CH3]>>[nH1:1]").
    The classifier matches these against _SMARTS_REACTION_MAP first, then
    falls back to human-readable keyword matching for BioTransformer labels.
    """
    label_lower = (reaction_label or "").lower()

    # ── Step 0: SMARTS-based reaction pattern matching ────────────────────────
    # This is the primary path for SyGMa metabolites — their labels ARE SMARTS.
    for frag, txn_type, enzyme in _SMARTS_REACTION_MAP:
        if frag in label_lower:
            # Refine CYP isoform from SMARTCyp scores if available
            if smartcyp_scores and "cyp" in enzyme.lower():
                best = min(smartcyp_scores, key=lambda s: s.get("ea", 999))
                iso  = _ISOFORM_TO_CLASS.get(best.get("isoform","").upper(), "")
                if iso:
                    return txn_type, iso
            # Refine isoform from parent structure if still generic
            if enzyme == "CYP" and parent_smiles:
                refined = _structural_demethylation_enzyme(parent_smiles)
                return txn_type, refined
            return txn_type, enzyme

    # ── Step 1: Human-readable keyword matching (BioTransformer / fallback) ───
    if "n-dealkylation" in label_lower or "o-dealkylation" in label_lower:
        enzyme = "CYP"
        if smartcyp_scores:
            best = min(smartcyp_scores, key=lambda s: s.get("ea", 999))
            iso  = _ISOFORM_TO_CLASS.get(best.get("isoform","").upper(), "")
            enzyme = iso if iso else enzyme
        elif parent_smiles:
            enzyme = _structural_demethylation_enzyme(parent_smiles)
        txn = "N-Dealkylation" if "n-dealkylation" in label_lower else "O-Dealkylation"
        return txn, enzyme

    if "methylation" in label_lower:
        if _is_true_methylation(label_lower):
            return "Methylation", "COMT / TPMT"
        else:
            enzyme = _structural_demethylation_enzyme(parent_smiles) if parent_smiles else "CYP"
            return "N/O-Dealkylation", enzyme

    for keyword, txn_type, enzyme in _TRANSFORMATION_MAP:
        if keyword in label_lower:
            if smartcyp_scores and enzyme in ("CYP", "CYP / FMO"):
                best    = min(smartcyp_scores, key=lambda s: s.get("ea", 999))
                isoform = _ISOFORM_TO_CLASS.get(best.get("isoform","").upper(), enzyme)
                return txn_type, isoform
            return txn_type, enzyme

    # ── Step 2: SMARTCyp scores fallback ─────────────────────────────────────
    if smartcyp_scores:
        best    = min(smartcyp_scores, key=lambda s: s.get("ea", 999))
        isoform = best.get("isoform", "").upper()
        rule    = best.get("rule", "").lower()
        enzyme  = _ISOFORM_TO_CLASS.get(isoform, isoform or "Unknown")
        for keyword, txn_type, _ in _TRANSFORMATION_MAP:
            if keyword in rule:
                return txn_type, enzyme
        if isoform.startswith("CYP"):
            return "CYP-mediated Oxidation", enzyme
        if isoform in ("UGT","SULT","FMO","AO","MAO","NAT","COMT","GST"):
            return f"{isoform}-mediated Reaction", isoform
        if isoform:
            return f"{isoform}-mediated", enzyme

    # ── Step 3: Phase-based fallback ─────────────────────────────────────────
    if phase == 1:
        return "Phase I Biotransformation", "CYP (predicted)"
    if phase == 2:
        return "Phase II Conjugation", "Conjugating Enzyme"
    if "sygma" in source_pipeline.lower():
        return "Metabolic Transformation", "SyGMa (rule-based)"
    return "Unknown Transformation", "Unknown"


def _classify_consensus(source_pipeline: str) -> tuple[bool, str]:
    """
    Determine consensus status from a pipe-delimited source_pipeline string.

    Rules
    -----
    Rule-based sources  : "sygma", "biotransformer"
    DL sources          : "dl"

    High Confidence requires ≥1 rule-based source AND ≥1 DL source.

    Returns
    -------
    (consensus_verified: bool, confidence_label: str)
    """
    sources = {s.strip().lower() for s in source_pipeline.split("|")}
    rule_sources = {"sygma", "biotransformer"}
    dl_sources   = {"dl"}

    has_rule = bool(sources & rule_sources)
    has_dl   = bool(sources & dl_sources)

    if has_rule and has_dl:
        return True, "High Confidence (Consensus Verified)"
    if has_rule and not has_dl:
        label = "Rule-Based Only" if len(sources & rule_sources) == 1 else "Rule-Based Consensus"
        return False, label
    if has_dl and not has_rule:
        return False, "DL Only"
    return False, "Single Source"


def _risk_colour(
    score: float,
    alpha_max: float,
    isoform: str = "",
    scheme: str = "risk",
) -> tuple[float, float, float, float]:
    """
    Map a normalised risk score [0, 1] to an RGBA colour tuple accepted by
    RDKit's ``DrawMolecule`` atom-colour dict (each channel in [0, 1]).

    risk scheme   : low  → pale yellow  →  deep red at high risk
    isoform scheme: colour determined by CYP isoform; brightness by score
    """
    score = max(0.0, min(1.0, score))
    alpha = alpha_max * score           # opacity scales with risk

    if scheme == "isoform" and isoform:
        r, g, b = _ISOFORM_RGB.get(isoform.upper(), _ISOFORM_RGB_DEFAULT)
        # Lighten for low scores: blend toward white
        blend = 1.0 - 0.6 * score
        r = r + (1.0 - r) * blend
        g = g + (1.0 - g) * blend
        b = b + (1.0 - b) * blend
        return (r, g, b, alpha)

    # ── Risk gradient: pale-yellow (low) → orange → deep-red (high) ──────────
    # Two-stop interpolation: [0, 0.5] yellow→orange; [0.5, 1.0] orange→red
    if score <= 0.5:
        t = score / 0.5            # 0→1 in first half
        r = 1.0
        g = 0.85 - 0.35 * t       # 0.85 → 0.50
        b = 0.20 - 0.20 * t       # 0.20 → 0.00
    else:
        t = (score - 0.5) / 0.5   # 0→1 in second half
        r = 1.0 - 0.15 * t        # 1.00 → 0.85
        g = 0.50 - 0.50 * t       # 0.50 → 0.00
        b = 0.0

    return (r, g, b, alpha)


def _draw_soft_spot_svg(
    smiles: str,
    atom_scores: list[AtomRiskScore],
    width: int,
    height: int,
    highlight_alpha_max: float,
    colour_scheme: str,
) -> str:
    """
    Core SVG rendering function.

    Steps
    -----
    1. Parse and sanitise the molecule; compute 2D coordinates.
    2. Build per-atom and per-bond colour maps from risk scores.
    3. Configure MolDraw2DSVG with Kekulé bonds and a clean white background.
    4. Draw with atom highlights, then post-process the raw SVG to embed a
       translucent circular glow element around each highlighted atom.
    5. Return the complete SVG string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES for rendering: {smiles!r}")

    Chem.SanitizeMol(mol)
    AllChem.Compute2DCoords(mol)

    # ── Build highlight maps ──────────────────────────────────────────────────
    # atom_idx → (R, G, B, A)  –  we split alpha out because RDKit's colour
    # dict expects (R, G, B) floats; alpha is applied via the SVG post-process.
    highlight_atoms: list[int] = []
    atom_colour_map: dict[int, tuple[float, float, float]] = {}
    atom_alpha_map:  dict[int, float] = {}
    atom_score_map:  dict[int, float] = {}

    for entry in atom_scores:
        idx = entry.atom_idx
        if idx >= mol.GetNumAtoms():
            continue                # silently skip out-of-range indices
        r, g, b, a = _risk_colour(
            entry.score,
            highlight_alpha_max,
            isoform=entry.isoform,
            scheme=colour_scheme,
        )
        highlight_atoms.append(idx)
        atom_colour_map[idx] = (r, g, b)
        atom_alpha_map[idx]  = a
        atom_score_map[idx]  = entry.score

    # Highlight bonds whose both endpoints are soft-spot atoms
    highlight_bonds: list[int] = []
    bond_colour_map: dict[int, tuple[float, float, float]] = {}
    soft_set = set(highlight_atoms)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in soft_set and j in soft_set:
            bond_idx = bond.GetIdx()
            highlight_bonds.append(bond_idx)
            # Average the two endpoint colours
            ri, gi, bi = atom_colour_map.get(i, (0.8, 0.8, 0.8))
            rj, gj, bj = atom_colour_map.get(j, (0.8, 0.8, 0.8))
            bond_colour_map[bond_idx] = (
                (ri + rj) / 2,
                (gi + gj) / 2,
                (bi + bj) / 2,
            )

    # ── Configure MolDraw2DSVG ────────────────────────────────────────────────
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    draw_opts = drawer.drawOptions()

    # Visual quality settings — only set attributes supported by this RDKit build.
    # highlightBondWidthMult was removed in RDKit 2023+; use atomHighlightsAreCircles
    # and the default bond highlight width instead.
    draw_opts.addStereoAnnotation = True
    draw_opts.addAtomIndices      = False
    draw_opts.bondLineWidth       = 2.0
    draw_opts.fillHighlights      = True
    draw_opts.padding             = 0.12
    # Apply each optional attribute only if the RDKit version supports it
    for attr, val in [
        ("highlightBondWidthMult",    8),
        ("atomHighlightsAreCircles",  True),
        ("continuousHighlight",       True),
    ]:
        try:
            setattr(draw_opts, attr, val)
        except AttributeError:
            pass
    draw_opts.useBWAtomPalette()

    # Draw molecule with highlight maps
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colour_map,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colour_map,
    )
    drawer.FinishDrawing()
    raw_svg: str = drawer.GetDrawingText()

    # ── Post-process: inject glow <circle> overlays ───────────────────────────
    # RDKit's highlight circles are solid; we add a second translucent ring
    # outside each atom whose opacity and radius scale with the risk score.
    # We do this by injecting SVG before the closing </svg> tag.
    glow_elements: list[str] = []

    conf = mol.GetConformer()
    # Derive pixel → Ångström scale from the drawn canvas dimensions.
    # MolDraw2DSVG exposes the coordinate mapping via GetDrawCoords().
    for idx in highlight_atoms:
        pt = drawer.GetDrawCoords(idx)
        px_x, px_y = pt.x, pt.y
        score      = atom_score_map.get(idx, 0.5)
        alpha      = atom_alpha_map.get(idx, 0.4)
        r_col, g_col, b_col = atom_colour_map.get(idx, (1.0, 0.5, 0.1))

        # Convert float colours → 0-255 integers for SVG rgb()
        r_int = int(round(r_col * 255))
        g_int = int(round(g_col * 255))
        b_int = int(round(b_col * 255))

        # Glow ring: radius proportional to risk (12–28 px), alpha proportional
        glow_r_inner = 12 + score * 8          # 12 → 20 px  (inner bright ring)
        glow_r_outer = glow_r_inner + 8        # outer diffuse halo
        glow_alpha   = round(alpha * 0.65, 3)  # slightly transparent inner ring
        halo_alpha   = round(alpha * 0.25, 3)  # very diffuse outer halo

        # Inner ring
        glow_elements.append(
            f'<circle cx="{px_x:.2f}" cy="{px_y:.2f}" r="{glow_r_inner:.1f}" '
            f'fill="rgba({r_int},{g_int},{b_int},{glow_alpha})" '
            f'stroke="rgba({r_int},{g_int},{b_int},{min(glow_alpha + 0.15, 1.0):.3f})" '
            f'stroke-width="1.5" />'
        )
        # Outer diffuse halo
        glow_elements.append(
            f'<circle cx="{px_x:.2f}" cy="{px_y:.2f}" r="{glow_r_outer:.1f}" '
            f'fill="rgba({r_int},{g_int},{b_int},{halo_alpha})" '
            f'stroke="none" />'
        )

    # Embed glow group just before </svg>
    if glow_elements:
        glow_block = (
            '\n  <g id="soft-spot-glows" opacity="1">\n    '
            + "\n    ".join(glow_elements)
            + "\n  </g>\n"
        )
        raw_svg = raw_svg.replace("</svg>", glow_block + "</svg>")

    # ── Enzyme colour key legend ─────────────────────────────────────────────
    # Collect which isoforms are actually present in this render
    isoforms_in_render: dict[str, tuple[float, float, float]] = {}
    for entry in atom_scores:
        if entry.atom_idx >= mol.GetNumAtoms():
            continue
        iso = (entry.isoform or "").strip().upper()
        if iso and iso not in isoforms_in_render:
            rc, gc, bc = atom_colour_map.get(entry.atom_idx, (0.7, 0.7, 0.7))
            isoforms_in_render[iso] = (rc, gc, bc)

    if isoforms_in_render:
        # Sort isoforms: CYPs first, then Phase II, then others
        CYP_ORDER = ["CYP3A4","CYP2D6","CYP2C9","CYP1A2","CYP2C19","CYP2E1"]
        P2_ORDER  = ["UGT","SULT","GST","NAT","COMT"]
        OX_ORDER  = ["FMO","AO","MAO"]
        def _iso_sort_key(k):
            if k in CYP_ORDER: return (0, CYP_ORDER.index(k))
            if k in P2_ORDER:  return (1, P2_ORDER.index(k))
            if k in OX_ORDER:  return (2, OX_ORDER.index(k))
            return (3, 0)
        sorted_isoforms = sorted(isoforms_in_render.keys(), key=_iso_sort_key)

        # Legend layout constants
        leg_x      = 8       # left margin
        leg_y      = 8       # top margin
        row_h      = 20      # height per legend row
        swatch_w   = 12      # colour swatch width
        swatch_h   = 12      # colour swatch height
        text_x     = leg_x + swatch_w + 6
        font_size  = 11
        n          = len(sorted_isoforms)
        box_w      = 130
        box_h      = 10 + n * row_h + 6  # dynamic height

        legend_parts: list[str] = [
            # Background panel
            f'<rect x="{leg_x - 4}" y="{leg_y - 4}" ',
            f'width="{box_w}" height="{box_h}" ',
            f'rx="5" fill="rgba(255,255,255,0.92)" ',
            f'stroke="#CBD5E0" stroke-width="1"/>',
            # "Enzyme Class" title
            f'<text x="{leg_x}" y="{leg_y + 9}" ',
            f'font-family="sans-serif" font-size="9" font-weight="bold" ',
            f'fill="#718096" letter-spacing="1">ENZYME CLASS</text>',
        ]

        for i, iso in enumerate(sorted_isoforms):
            r_c, g_c, b_c = isoforms_in_render[iso]
            ri3 = int(round(r_c * 255))
            gi3 = int(round(g_c * 255))
            bi3 = int(round(b_c * 255))
            row_y = leg_y + 14 + i * row_h

            # Colour swatch with rounded corners
            legend_parts.append(
                f'<rect x="{leg_x}" y="{row_y}" ',
            )
            legend_parts.append(
                f'width="{swatch_w}" height="{swatch_h}" ',
            )
            legend_parts.append(
                f'rx="3" fill="rgb({ri3},{gi3},{bi3})" ',
            )
            legend_parts.append(
                f'stroke="rgba({ri3},{gi3},{bi3},0.6)" stroke-width="1"/>',
            )
            # Isoform label
            legend_parts.append(
                f'<text x="{text_x}" y="{row_y + 9}" ',
            )
            legend_parts.append(
                f'font-family="sans-serif" font-size="{font_size}" ',
            )
            legend_parts.append(
                f'fill="#2D3748">{iso}</text>',
            )

        legend_block = (
            '<g id="enzyme-colour-key">\n  '
            + "\n  ".join(legend_parts)
            + "\n</g>\n"
        )
        raw_svg = raw_svg.replace("</svg>", legend_block + "</svg>")

    return raw_svg


# ---------------------------------------------------------------------------
# Middleware: request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Attach X-Process-Time header to every response."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed  = time.perf_counter() - t0
    response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health_check() -> dict[str, str]:
    """Liveness probe — returns 200 if the server is running."""
    return {"status": "ok", "version": app.version}


# ── POST /predict ─────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
    summary="Run full metabolite prediction pipeline",
    response_description=(
        "Structured JSON with parent HRMS metrics, all predicted metabolites "
        "annotated with masses / adducts / soft-spot atoms / consensus tags, "
        "and pipeline statistics."
    ),
)
async def predict(body: PredictRequest) -> PredictResponse:
    """
    **Prediction workflow**

    1. Validate and canonicalise the parent SMILES (done by Pydantic validator).
    2. Delegate to `aggregate_metabolism()` which runs all four pipelines in
       sequence, deduplicates on InChIKey, and returns an ensemble-scored matrix.
    3. Apply the **Ensemble Consensus Filter**: tag any metabolite confirmed by
       both a rule-based path (SyGMa / BioTransformer) *and* a DL model
       (MetaTrans / Meta-Predictor) as *"High Confidence (Consensus Verified)"*.
    4. Enrich each entry with molecular formula and adduct model fields.
    5. Return the complete `PredictResponse` payload.
    """
    t0 = time.perf_counter()
    log.info("POST /predict  smiles=%r  pipelines=[sygma=%s bt=%s dl=%s sc=%s]",
             body.smiles, body.run_sygma, body.run_biotransformer,
             body.run_dl, body.run_smartcyp)

    # ── Run all pipelines ─────────────────────────────────────────────────────
    try:
        raw: dict[str, Any] = aggregate_metabolism(
            parent_smiles         = body.smiles,
            run_sygma             = body.run_sygma,
            run_biotransformer    = body.run_biotransformer,
            run_dl                = body.run_dl,
            run_smartcyp          = body.run_smartcyp,
            biotransformer_jar    = body.biotransformer_jar,
            biotransformer_type   = body.biotransformer_type,
            dl_model_dir          = body.dl_model_dir,
            dl_device             = body.dl_device,
            sygma_phase1_cycles   = body.sygma_phase1_cycles,
            sygma_phase2_cycles   = body.sygma_phase2_cycles,
            smartcyp_ea_cutoff    = body.smartcyp_ea_cutoff,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.exception("aggregate_metabolism raised an unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {exc}",
        ) from exc

    # ── Build parent metrics ──────────────────────────────────────────────────
    raw_parent = raw["parent"]
    parent_formula = _molecular_formula(raw_parent["smiles"])

    # ── Apply post-processing improvements (UGT re-scoring, sequential) ──────
    try:
        raw["metabolites"] = apply_all_improvements(
            metabolite_list = raw["metabolites"],
            parent_smiles   = raw_parent["smiles"],
            run_sequential  = body.run_sequential,
            run_ndealk      = body.run_ndealk,
            run_reduction   = body.run_reduction,
            verbose         = False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("apply_all_improvements failed (non-fatal): %s", exc)

    # -- DMPK-empirical prioritisation + regioisomer capping --------------------
    # Attach the human-readable transformation_type so the prioritiser can tell
    # hindered aromatic hydroxylations (downgraded) from accessible aliphatic
    # oxidations, then cap regioisomers and rescore by mass-spec relevance.
    if body.prioritize:
        try:
            for _m in raw["metabolites"]:
                _sc = [{"rule": s.get("rule", ""), "isoform": s.get("isoform", ""),
                        "ea": s.get("ea", 0.0)} for s in _m.get("smartcyp_scores", [])]
                _ttype, _ = _classify_transformation(
                    _m.get("reaction_label", ""), _sc,
                    _m.get("source_pipeline", ""), _m.get("phase", 0),
                    raw["parent"]["smiles"],
                )
                _m["transformation_type"] = _ttype
            raw["metabolites"] = prioritize_predictions(raw["metabolites"])
        except Exception as exc:  # noqa: BLE001
            log.warning("prioritize_predictions failed (non-fatal): %s", exc)

    parent_out = ParentMetrics(
        smiles           = raw_parent["smiles"],
        neutral_mass     = raw_parent["neutral_mass"],
        molecular_formula= parent_formula,
        adducts          = HRMSAdducts(
            mplus_h  = raw_parent["adduct_mplus_h"],
            mminus_h = raw_parent["adduct_mminus_h"],
        ),
    )

    # ── Build metabolite list with consensus filter ───────────────────────────
    # `metabolites_out` holds the top_n ranked entries shown in the report.
    # `all_metabolites_out` holds every predicted metabolite and is populated
    # only when return_all=True — it is the full LC-HRMS target list.
    metabolites_out: list[MetaboliteEntry] = []
    all_metabolites_out: list[MetaboliteEntry] = []
    consensus_count = 0

    for rank, m in enumerate(raw["metabolites"], start=1):
        consensus_verified, confidence_label = _classify_consensus(
            m["source_pipeline"]
        )
        if consensus_verified:
            consensus_count += 1

        # Validate SMARTCyp score entries (gracefully skip malformed dicts)
        sc_scores: list[SMARTCypMatch] = []
        for sc in m.get("smartcyp_scores", []):
            try:
                sc_scores.append(
                    SMARTCypMatch(
                        rule    = sc.get("rule", ""),
                        isoform = sc.get("isoform", ""),
                        ea      = float(sc.get("ea", 0.0)),
                        atoms   = sc.get("atoms", []),
                    )
                )
            except Exception:  # noqa: BLE001
                pass

        txn_type, resp_enzyme = _classify_transformation(
            m.get("reaction_label", ""),
            [{"rule": s.rule, "isoform": s.isoform, "ea": s.ea} for s in sc_scores],
            m.get("source_pipeline", ""),
            m.get("phase", 0),
            raw["parent"]["smiles"],
        )
        entry = MetaboliteEntry(
            rank               = rank,
            smiles_canonical   = m["smiles_canonical"],
            inchikey           = m["inchikey"],
            molecular_formula  = _molecular_formula(m["smiles_canonical"]),
            neutral_mass       = m["neutral_mass"],
            delta_mass         = (m["delta_mass"] if isinstance(m["delta_mass"], str)
                                  else f"{float(m['delta_mass']):+.4f}"),
            adducts            = HRMSAdducts(
                mplus_h  = m["adduct_mplus_h"],
                mminus_h = m["adduct_mminus_h"],
            ),
            source_pipeline    = m["source_pipeline"],
            reaction_label     = m.get("reaction_label", ""),
            phase              = m.get("phase", 0),
            soft_spot_atoms    = m.get("soft_spot_atoms", []),
            smartcyp_scores    = sc_scores,
            dl_confidence      = m.get("dl_confidence", 0.0),
            ensemble_score     = m.get("ensemble_score", 0.0),
            consensus_verified  = consensus_verified,
            confidence_label    = confidence_label,
            transformation_type = txn_type,
            responsible_enzyme  = resp_enzyme,
        )
        # Always populate all_metabolites_out (used when return_all=True)
        all_metabolites_out.append(entry)
        # Only add to ranked list if within top_n
        if rank <= body.top_n:
            metabolites_out.append(entry)

    # ── Pipeline stats ────────────────────────────────────────────────────────
    raw_stats = raw["pipeline_stats"]
    elapsed = round(time.perf_counter() - t0, 4)
    stats_out = PipelineStats(
        sygma_count          = raw_stats["sygma_count"],
        biotransformer_count = raw_stats["biotransformer_count"],
        dl_count             = raw_stats["dl_count"],
        smartcyp_count       = raw_stats["smartcyp_count"],
        total_after_dedup    = raw_stats["total_after_dedup"],
        consensus_count      = consensus_count,
        elapsed_seconds      = elapsed,
    )

    log.info(
        "POST /predict complete — %d metabolites (%d consensus) in %.3fs",
        stats_out.total_after_dedup,
        consensus_count,
        elapsed,
    )

    return PredictResponse(
        parent             = parent_out,
        metabolites        = metabolites_out,
        all_metabolites    = all_metabolites_out if body.return_all else [],
        soft_spot_summary  = SoftSpotSummary(**raw["soft_spot_summary"]),
        pipeline_stats     = stats_out,
    )


# ── POST /render-soft-spots ───────────────────────────────────────────────────

@app.post(
    "/render-soft-spots",
    tags=["visualization"],
    summary="Render 2D molecule SVG with risk-proportional soft-spot highlights",
    responses={
        200: {
            "content": {"image/svg+xml": {}},
            "description": "SVG string with glow overlays at vulnerable atom positions.",
        },
        422: {"description": "Invalid SMILES or malformed atom scores."},
    },
)
async def render_soft_spots(body: SoftSpotRenderRequest) -> Response:
    """
    **SVG rendering workflow**

    1. Parse and sanitise the SMILES; compute 2D coordinates with RDKit.
    2. Map each `AtomRiskScore` entry to an RGBA colour using either the
       *risk* gradient (pale-yellow → deep-red) or fixed *isoform* colours.
    3. Call `MolDraw2DSVG.DrawMolecule()` with the per-atom colour maps.
    4. Post-process the raw SVG: inject a translucent `<circle>` glow at each
       vulnerable atom position with radius and opacity proportional to score.
    5. Append a compact legend listing the top-5 annotated atoms.
    6. Return as `Response(content=..., media_type="image/svg+xml")`.
    """
    log.info(
        "POST /render-soft-spots  smiles=%r  n_scores=%d  scheme=%s",
        body.smiles, len(body.atom_scores), body.colour_scheme,
    )

    try:
        svg_string = _draw_soft_spot_svg(
            smiles              = body.smiles,
            atom_scores         = body.atom_scores,
            width               = body.width,
            height              = body.height,
            highlight_alpha_max = body.highlight_alpha_max,
            colour_scheme       = body.colour_scheme,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        log.exception("SVG rendering failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rendering error: {exc}",
        ) from exc

    return Response(content=svg_string, media_type="image/svg+xml")


# ---------------------------------------------------------------------------
# Global exception handler — returns structured JSON for unhandled errors
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error":   type(exc).__name__,
            "detail":  str(exc),
            "path":    str(request.url),
        },
    )


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------
# Run with:  uvicorn app.main:app --reload --port 8000
# Or:        python app/main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
