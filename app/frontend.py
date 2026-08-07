"""
app/frontend.py
===============
HRMS Metabolite Predictor — Streamlit Frontend

Clean light pharmaceutical SaaS interface — white base with blue/green accents.

Layout
------
  Header bar      — branding + server status indicator
  Input panel     — Ketcher chemical sketcher + SMILES text fallback
  Pipeline config — expandable sidebar-style settings
  Dashboard       — 3-column analytical split:
                      Left   : KPI metric cards
                      Centre : RDKit soft-spot SVG map
                      Right  : LC-MS screening data table
  Footer          — download controls + pipeline stats

Run
---
    streamlit run app/frontend.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import Any

# `streamlit run app/frontend.py` puts the app/ folder on sys.path, but NOT the
# repo root. Add the repo root so the in-process engine import (`from app.main
# import app`) and the vendored `sygma` package (at repo-root ./sygma) resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import requests
import streamlit as st
from streamlit_ketcher import st_ketcher

try:
    from app.cdd_client import fetch_smiles_by_name, CDDError  # package run
except Exception:  # noqa: BLE001
    from cdd_client import fetch_smiles_by_name, CDDError       # `streamlit run app/frontend.py`

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND_URL   = os.environ.get("HRMS_BACKEND_URL", "http://localhost:8080")
_BACKEND_PORT = BACKEND_URL.rsplit(":", 1)[-1].split("/")[0] or "8080"
PREDICT_URL   = f"{BACKEND_URL}/predict"
SOFT_SPOT_URL = f"{BACKEND_URL}/render-soft-spots"
HEALTH_URL    = f"{BACKEND_URL}/health"

# ---------------------------------------------------------------------------
# Transport: use a real HTTP backend when one is reachable, otherwise run the
# FastAPI engine IN-PROCESS (e.g. on Streamlit Community Cloud, where only the
# Streamlit process exists). In-process mode reuses app/main.py verbatim via a
# Starlette TestClient -- no second server, no ports.
# ---------------------------------------------------------------------------
_MODE = None            # "http" | "inprocess" | None (offline)
_IP_CLIENT = None


def _get_inprocess_client():
    global _IP_CLIENT
    if _IP_CLIENT is None:
        from fastapi.testclient import TestClient
        from app.main import app as _fastapi_app
        _IP_CLIENT = TestClient(_fastapi_app, raise_server_exceptions=False)
    return _IP_CLIENT


def _path_of(url: str) -> str:
    if "://" in url:
        rest = url.split("://", 1)[1]
        return "/" + rest.split("/", 1)[1] if "/" in rest else "/"
    return url


def _http_post(url, json=None, timeout=None):
    if _MODE == "inprocess":
        return _get_inprocess_client().post(_path_of(url), json=json)
    return requests.post(url, json=json, timeout=timeout)


def _http_get(url, timeout=None):
    if _MODE == "inprocess":
        return _get_inprocess_client().get(_path_of(url))
    return requests.get(url, timeout=timeout)

COLOUR_SCHEMES = {"Enzyme class (recommended)": "isoform", "Risk gradient": "risk"}

# ---------------------------------------------------------------------------
# Page configuration  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HRMS Metabolite Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "HRMS Predictor v1.0 — Multi-engine metabolite prediction suite",
    },
)

# ---------------------------------------------------------------------------
# CSS — clean light pharmaceutical theme (white + blue/green accents)
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════════════════
   SIDEBAR — fixed width, no label clipping
   ══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    min-width: 270px !important;
    max-width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 270px !important;
    padding: 1rem 0.75rem !important;
}

/* ══════════════════════════════════════════════════════
   EXPANDER — clean summary row, no arrow/text overlap
   ══════════════════════════════════════════════════════ */
[data-testid="stExpander"] details > summary {
    list-style: none !important;
    display: flex !important;
    align-items: center !important;
    padding: 0.5rem 0.6rem !important;
    cursor: pointer !important;
    overflow: visible !important;
    white-space: normal !important;
    gap: 0 !important;
}
[data-testid="stExpander"] details > summary::-webkit-details-marker,
[data-testid="stExpander"] details > summary::marker { display: none !important; }

[data-testid="stExpander"] details > summary p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    color: #3A3E45 !important;
    white-space: normal !important;
    overflow: visible !important;
    flex: 1 !important;
    line-height: 1.3 !important;
}
[data-testid="stExpander"] details > summary svg {
    width: 14px !important;
    height: 14px !important;
    flex-shrink: 0 !important;
    margin-right: 5px !important;
}

/* ══════════════════════════════════════════════════════
   TABS — pill style, NO red/orange underline bleed
   ══════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
    height: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #EAF0FF !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 3px !important;
    border: 1px solid #D8D8D8 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    color: #6B7078 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.32rem 0.85rem !important;
    transition: all 0.15s ease !important;
    outline: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0046FF !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(29,31,36,0.18) !important;
}

/* ══════════════════════════════════════════════════════
   DISABLED PIPELINE CARDS
   ══════════════════════════════════════════════════════ */
.pipeline-unavailable {
    background: #F5F6F7;
    border: 1px solid #D8D8D8;
    border-radius: 7px;
    padding: 0.42rem 0.6rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.pipeline-unavailable-label {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 0.82rem;
    color: #9AA0A6;
    text-decoration: line-through;
    flex: 1;
}
.pipeline-unavailable-badge {
    font-size: 0.62rem;
    color: #9AA0A6;
    background: #EEEFF1;
    border: 1px solid #D8D8D8;
    border-radius: 20px;
    padding: 1px 6px;
    white-space: nowrap;
}

:root {
    --bg-base:        #F5F6F7;
    --bg-surface:     #FFFFFF;
    --bg-elevated:    #EAF0FF;
    --bg-card:        #FFFFFF;
    --bg-input:       #FFFFFF;
    --bg-sidebar:     #F4F6FB;

    --border:         #D8D8D8;
    --border-subtle:  #E6EAF2;
    --border-accent:  #0046FF;

    --accent-primary: #0046FF;
    --accent-blue:    #0046FF;
    --accent-teal:    #083651;
    --accent-green:   #00C885;
    --accent-sky:     #0046FF;
    --accent-mint:    #00C885;
    --accent-amber:   #D97706;
    --accent-red:     #DC2626;

    --text-primary:   #1D1F24;
    --text-secondary: #3A3E45;
    --text-muted:     #6B7078;
    --text-light:     #9AA0A6;

    --font-display:   "Helvetica Neue", Helvetica, Arial, sans-serif;
    --font-body:      "Helvetica Neue", Helvetica, Arial, sans-serif;
    --font-mono:      'DM Mono', monospace;

    --radius-sm:      6px;
    --radius-md:      10px;
    --radius-lg:      16px;
    --radius-xl:      24px;

    --shadow-card:    0 1px 8px rgba(29,31,36,0.10), 0 1px 3px rgba(29,31,36,0.06);
    --shadow-hover:   0 4px 20px rgba(29,31,36,0.15), 0 1px 6px rgba(29,31,36,0.08);
    --shadow-glow:    0 0 20px rgba(0,70,255,0.20);
}

/* ── Global resets ──────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; max-width: 100% !important; }

/* ── Typography ─────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

p, li, span, div, label {
    font-family: var(--font-body) !important;
    color: var(--text-primary);
}

code, pre, [data-testid="stCode"] {
    font-family: var(--font-mono) !important;
    background: var(--bg-elevated) !important;
    color: var(--accent-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Input fields ───────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: var(--bg-input) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(0,70,255,0.15) !important;
    outline: none !important;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0046FF 0%, #0046FF 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(29,31,36,0.30) !important;
    cursor: pointer !important;
}

[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #3A3E45 0%, #0046FF 100%) !important;
    box-shadow: 0 4px 18px rgba(29,31,36,0.45) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

[data-testid="stDownloadButton"] > button {
    background: #FFFFFF !important;
    border: 1.5px solid var(--accent-teal) !important;
    color: var(--accent-teal) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: rgba(13,148,136,0.06) !important;
    box-shadow: 0 2px 12px rgba(13,148,136,0.20) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs: now handled above in the top CSS block ───────────────────── */

/* ── Selectbox ──────────────────────────────────────────────────────── */
[data-baseweb="select"] > div:first-child {
    background-color: var(--bg-input) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}

/* ── Dataframe ──────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-card) !important;
}

[data-testid="stDataFrame"] th {
    background-color: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    border-bottom: 1.5px solid var(--border) !important;
}

[data-testid="stDataFrame"] td {
    background-color: #FFFFFF !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ── Expander card style ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 0.5rem !important;
    box-shadow: var(--shadow-card) !important;
    overflow: visible !important;
}

/* ── Checkbox ───────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label span {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-light); }

hr { border-color: var(--border) !important; opacity: 0.8 !important; }

/* ── App header ─────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 1.2rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 1.6rem;
}

.app-logo-mark {
    font-family: var(--font-display);
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
}

.app-logo-mark span { color: var(--accent-blue); }

.app-logo-sub {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    color: var(--text-muted);
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-top: 2px;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

.status-online  { background: #00C885; box-shadow: 0 0 6px rgba(5,150,105,0.5); }
.status-offline { background: #DC2626; box-shadow: 0 0 6px rgba(220,38,38,0.5); }
.status-label { font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted); vertical-align: middle; }

/* ── KPI cards ──────────────────────────────────────────────────────── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem 1rem;
    box-shadow: var(--shadow-card);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
    margin-bottom: 0.85rem;
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-line, var(--accent-blue));
    border-radius: 3px 3px 0 0;
}

.kpi-card:hover {
    border-color: var(--accent-blue);
    box-shadow: var(--shadow-hover);
}

.kpi-label {
    font-family: var(--font-display);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-family: var(--font-mono);
    font-size: 1.7rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.kpi-sub {
    font-family: var(--font-body);
    font-size: 0.71rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Section header ─────────────────────────────────────────────────── */
.section-header {
    font-family: var(--font-display);
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    padding-bottom: 0.45rem;
    border-bottom: 1.5px solid var(--border-subtle);
}

/* ── SVG frame ──────────────────────────────────────────────────────── */
.svg-frame {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem;
    box-shadow: var(--shadow-card);
    text-align: center;
    min-height: 320px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.svg-frame svg { max-width: 100%; height: auto; border-radius: var(--radius-sm); }

/* ── Pipeline badge pills ───────────────────────────────────────────── */
.pipe-badge {
    display: inline-block;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    padding: 2px 8px;
    margin: 1px 2px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-sygma          { background: #CCFBF1; color: #06293D; border: 1px solid #99F6E4; }
.badge-biotransformer { background: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE; }
.badge-dl             { background: #FCE7F3; color: #9D174D; border: 1px solid #FBCFE8; }
.badge-smartcyp       { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }

/* ── Enzyme isoform legend badges ───────────────────────────────────── */
.isoform-UGT   { background: #D6F7EC; color: #06523A; border: 1px solid #6FE3BF; }
.isoform-SULT  { background: #DCE6FF; color: #0037CC; border: 1px solid #9DB4FF; }
.isoform-FMO   { background: #FEF3C7; color: #78350F; border: 1px solid #FCD34D; }
.isoform-AO    { background: #FCE7F3; color: #831843; border: 1px solid #F9A8D4; }
.isoform-MAO   { background: #EDE9FE; color: #4C1D95; border: 1px solid #C4B5FD; }
.isoform-NAT   { background: #E9FBF4; color: #083D2A; border: 1px solid #6FE3BF; }
.isoform-COMT  { background: #FFF7ED; color: #7C2D12; border: 1px solid #FDBA74; }
.isoform-GST   { background: #E9FBF4; color: #06523A; border: 1px solid #6FE3BF; }
.isoform-CYP3A4{ background: #FEF2F2; color: #7F1D1D; border: 1px solid #FCA5A5; }
.isoform-CYP2D6{ background: #EFF3FF; color: #0037CC; border: 1px solid #9DB4FF; }
.isoform-CYP2C9{ background: #FFF7ED; color: #7C2D12; border: 1px solid #FDBA74; }

/* ── Consensus chips ────────────────────────────────────────────────── */
.consensus-high {
    display: inline-block;
    background: #D6F7EC;
    color: #06523A;
    border: 1px solid #6FE3BF;
    border-radius: 20px;
    font-size: 0.70rem;
    font-weight: 600;
    padding: 2px 10px;
    font-family: var(--font-display);
    letter-spacing: 0.03em;
}

.consensus-rule {
    display: inline-block;
    background: #DCE6FF;
    color: #0037CC;
    border: 1px solid #9DB4FF;
    border-radius: 20px;
    font-size: 0.70rem;
    padding: 2px 10px;
    font-family: var(--font-display);
}

.consensus-dl {
    display: inline-block;
    background: #FAE8FF;
    color: #701A75;
    border: 1px solid #E879F9;
    border-radius: 20px;
    font-size: 0.70rem;
    padding: 2px 10px;
    font-family: var(--font-display);
}

.consensus-single {
    display: inline-block;
    background: #F1F2F4;
    color: #4A4E57;
    border: 1px solid #CFD0D2;
    border-radius: 20px;
    font-size: 0.70rem;
    padding: 2px 10px;
    font-family: var(--font-display);
}

/* ── Empty state ────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
    font-family: var(--font-body);
}

.empty-icon { font-size: 2.8rem; margin-bottom: 0.75rem; opacity: 0.4; }
.empty-text { font-size: 0.88rem; line-height: 1.65; }

/* ── Ketcher iframe ─────────────────────────────────────────────────── */
[data-testid="stIframe"] iframe {
    border-radius: var(--radius-lg) !important;
    border: 1.5px solid var(--border) !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _check_backend_health() -> bool:
    global _MODE
    force_ip = os.environ.get("HRMS_INPROCESS", "").strip().lower() in ("1", "true", "yes")
    if not force_ip:
        try:
            r = requests.get(HEALTH_URL, timeout=2.5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                _MODE = "http"
                return True
        except Exception:
            pass
    # No reachable HTTP backend -> run the full engine in-process.
    try:
        r = _get_inprocess_client().get("/health")
        if r.status_code == 200 and r.json().get("status") == "ok":
            _MODE = "inprocess"
            return True
    except Exception:
        import traceback
        globals()["_IP_ERR"] = traceback.format_exc()
        traceback.print_exc()
    _MODE = None
    return False


def _pipeline_badge(name: str) -> str:
    cls = {
        "sygma":          "badge-sygma",
        "biotransformer": "badge-biotransformer",
        "dl":             "badge-dl",
        "smartcyp":       "badge-smartcyp",
    }.get(name.strip().lower(), "badge-smartcyp")
    return f'<span class="pipe-badge {cls}">{name.strip()}</span>'


def _consensus_chip(label: str) -> str:
    ll = label.lower()
    if "consensus verified" in ll:
        return f'<span class="consensus-high">✦ {label}</span>'
    elif "rule-based" in ll:
        return f'<span class="consensus-rule">{label}</span>'
    elif "dl only" in ll:
        return f'<span class="consensus-dl">{label}</span>'
    return f'<span class="consensus-single">{label}</span>'


def _build_display_df(metabolites: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in metabolites:
        adducts = m.get("adducts", {})
        rows.append({
            "Rank":               m.get("rank", 0),
            "Structure (SMILES)": m.get("smiles_canonical", ""),
            "Transformation":     m.get("transformation_type", ""),
            "Enzyme":             m.get("responsible_enzyme", ""),
            "Source Engine":      m.get("source_pipeline", ""),
            "Formula":            m.get("molecular_formula", ""),
            "Neutral Mass (Da)":  m.get("neutral_mass", 0.0),
            "Δ m/z":              m.get("delta_mass", ""),
            "[M+H]⁺ m/z":        adducts.get("mplus_h", 0.0),
            "[M-H]⁻ m/z":        adducts.get("mminus_h", 0.0),
            "Phase":              m.get("phase", 0),
            "DL Confidence":      m.get("dl_confidence", 0.0),
            "Ensemble Score":     m.get("ensemble_score", 0.0),
            "Consensus Status":   m.get("confidence_label", ""),
            "InChIKey":           m.get("inchikey", ""),
            "Soft-Spot Atoms":    str(m.get("soft_spot_atoms", [])),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    def _delta_abs(v: str) -> float:
        try:
            return abs(float(str(v).replace("+", "").replace("-", "")))
        except (ValueError, AttributeError):
            return 999.0

    df["_delta_abs"] = df["Δ m/z"].apply(_delta_abs)
    df = df.sort_values(
        by=["_delta_abs", "Ensemble Score"],
        ascending=[True, False],
    ).drop(columns=["_delta_abs"]).reset_index(drop=True)
    return df


def _build_excel_bytes(df: pd.DataFrame, parent: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Metabolites", index=False)
        parent_adducts = parent.get("adducts", {})
        pd.DataFrame([
            ["Parent SMILES",             parent.get("smiles", "")],
            ["Molecular Formula",         parent.get("molecular_formula", "")],
            ["Neutral Monoisotopic Mass",  parent.get("neutral_mass", "")],
            ["[M+H]+ m/z",               parent_adducts.get("mplus_h", "")],
            ["[M-H]- m/z",               parent_adducts.get("mminus_h", "")],
        ], columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Parent Metrics", index=False
        )
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(
                    (len(str(cell.value)) for cell in col if cell.value is not None),
                    default=8,
                )
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def _build_pdf_bytes(df: pd.DataFrame, pm: pd.DataFrame, compound_name: str,
                     top_n: int, atom_scores_json: str) -> bytes:
    """Build the professional PDF report (cover + colour-coded soft-spot map +
    top-N metabolite cards + summary table) and return the bytes. Cached so it
    only regenerates when inputs change."""
    from generate_report import build_report_from_frames
    atom_scores = json.loads(atom_scores_json) if atom_scores_json else None
    return build_report_from_frames(
        df, pm, compound_name=compound_name or "Predicted Compound",
        top_n=int(top_n), atom_scores=atom_scores,
    )


def _fetch_soft_spot_svg(
    smiles: str,
    metabolites: list[dict[str, Any]],
    scheme: str,
    alpha_max: float,
    soft_spot_summary: dict[str, Any] | None = None,
    width: int = 560,
    height: int = 380,
) -> str | None:
    # Collect max score and ALL isoforms per atom
    atom_max_score: dict[int, float]      = {}
    atom_isoforms:  dict[int, set[str]]   = {}

    for m in metabolites:
        escore = m.get("ensemble_score", 0.0)
        for atom_idx in m.get("soft_spot_atoms", []):
            atom_max_score[atom_idx] = max(atom_max_score.get(atom_idx, 0.0), escore)
        for sc in m.get("smartcyp_scores", []):
            iso = sc.get("isoform", "").strip()
            if not iso:
                continue
            for atom_idx in sc.get("atoms", []):
                if atom_idx not in atom_isoforms:
                    atom_isoforms[atom_idx] = set()
                atom_isoforms[atom_idx].add(iso)
                # Also ensure the atom appears in score map
                atom_max_score[atom_idx] = max(atom_max_score.get(atom_idx, 0.0), escore)

    # Build the per-atom payload from the metabolites shown in the ranked table.
    payload_from_mets = bool(atom_max_score)

    # Fallback: the authoritative soft-spot scores are computed by the backend
    # over the FULL metabolite set before any prioritisation/capping, and live
    # in soft_spot_summary["atom_scores"]. The SMARTCyp parent record that
    # carries per-metabolite soft_spot_atoms can be ranked out of the shown
    # top-N, leaving atom_max_score empty even though soft spots exist — so we
    # rebuild from the summary whenever the metabolite-derived map is empty.
    if not payload_from_mets and soft_spot_summary:
        for entry in soft_spot_summary.get("atom_scores", []):
            idx = entry.get("atom_idx")
            if idx is None:
                continue
            sc = float(entry.get("score", 0.0))
            atom_max_score[idx] = max(atom_max_score.get(idx, 0.0), sc)
            iso = (entry.get("isoform") or "").strip()
            if iso:
                atom_isoforms.setdefault(idx, set()).add(iso)
        # Last resort: top_atoms with a uniform mid score (no isoform info).
        if not atom_max_score:
            for idx in soft_spot_summary.get("top_atoms", []):
                atom_max_score[idx] = 0.5

    if not atom_max_score:
        return None

    max_score = max(atom_max_score.values()) or 1.0

    # For each atom: pick the highest-priority isoform to display
    # Priority order: CYP3A4 > CYP2D6 > CYP2C9 > UGT > SULT > FMO > AO > MAO > others
    PRIORITY = ["CYP3A4","CYP2D6","CYP2C9","CYP1A2","CYP2C19","CYP2E1",
                "UGT","SULT","FMO","AO","MAO","NAT","COMT","GST"]

    atom_scores_payload = []
    for idx, score in atom_max_score.items():
        isos = atom_isoforms.get(idx, set())
        # Pick the top-priority isoform that matched this atom
        chosen_iso = ""
        for p in PRIORITY:
            if p in isos:
                chosen_iso = p
                break
        if not chosen_iso and isos:
            chosen_iso = sorted(isos)[0]  # fallback: alphabetical
        atom_scores_payload.append({
            "atom_idx": idx,
            "score":    round(score / max_score, 4),
            "isoform":  chosen_iso,
        })

    try:
        r = _http_post(SOFT_SPOT_URL, json={
            "smiles":              smiles,
            "atom_scores":         atom_scores_payload,
            "width":               width,
            "height":              height,
            "highlight_alpha_max": alpha_max,
            "colour_scheme":       scheme,
        }, timeout=20)
        if r.status_code == 200:
            return r.text
        st.error(f"SVG render error {r.status_code}: {r.text[:200]}")
        return None
    except Exception as exc:
        st.error(f"Cannot reach render endpoint: {exc}")
        return None


# ---------------------------------------------------------------------------
# UI component renderers
# ---------------------------------------------------------------------------

def _render_header(backend_online: bool) -> None:
    status_cls  = "status-online"  if backend_online else "status-offline"
    status_text = "Backend online" if backend_online else "Backend offline"
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-logo-mark">HRMS<span>·</span>Predict</div>
                <div class="app-logo-sub">Multi-engine metabolite prediction suite</div>
            </div>
            <div>
                <span class="status-dot {status_cls}"></span>
                <span class="status-label">{status_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_cards(
    parent: dict[str, Any],
    stats:  dict[str, Any],
    metabolites: list[dict[str, Any]],
) -> None:
    neutral_mass = parent.get("neutral_mass", 0.0)
    total        = stats.get("total_after_dedup", 0)
    consensus    = stats.get("consensus_count", 0)
    elapsed      = stats.get("elapsed_seconds", 0.0)
    max_vuln     = max(
        (m["ensemble_score"] for m in metabolites if m.get("soft_spot_atoms")),
        default=0.0,
    )

    cards = [
        {"label": "Parent Monoisotopic Mass",
         "value": f"{neutral_mass:.4f}",
         "sub":   f"Da  ·  {parent.get('molecular_formula', '')}",
         "line":  "#0046FF"},
        {"label": "Metabolites Generated",
         "value": str(total),
         "sub":   "after cross-engine deduplication",
         "line":  "#083651"},
        {"label": "Consensus Verified",
         "value": str(consensus),
         "sub":   f"{(consensus/total*100):.0f}% of total" if total else "—",
         "line":  "#00C885"},
        {"label": "Max Soft-Spot Score",
         "value": f"{max_vuln:.3f}",
         "sub":   f"normalised ensemble  ·  {elapsed:.2f}s",
         "line":  "#D97706"},
    ]

    for card in cards:
        st.markdown(
            f"""
            <div class="kpi-card" style="--accent-line:{card['line']};">
                <div class="kpi-label">{card['label']}</div>
                <div class="kpi-value">{card['value']}</div>
                <div class="kpi-sub">{card['sub']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_svg_panel(
    smiles: str,
    metabolites: list[dict[str, Any]],
    scheme: str,
    alpha_max: float,
    soft_spot_summary: dict[str, Any] | None = None,
) -> None:
    st.markdown('<div class="section-header">Metabolic Soft-Spot Map</div>',
                unsafe_allow_html=True)

    with st.spinner("Rendering atom vulnerability map…"):
        svg = _fetch_soft_spot_svg(smiles, metabolites, scheme, alpha_max,
                                   soft_spot_summary=soft_spot_summary)

    if svg:
        st.markdown(f'<div class="svg-frame">{svg}</div>', unsafe_allow_html=True)

        # Collect isoforms and their atom counts for the caption
        iso_counts: dict[str, int] = {}
        for m in metabolites:
            for sc in m.get("smartcyp_scores", []):
                iso = sc.get("isoform", "").strip()
                if iso:
                    iso_counts[iso] = iso_counts.get(iso, 0) + len(sc.get("atoms", []))
        if not iso_counts and soft_spot_summary:
            for entry in soft_spot_summary.get("atom_scores", []):
                iso = (entry.get("isoform") or "").strip()
                if iso:
                    iso_counts[iso] = iso_counts.get(iso, 0) + 1

        if iso_counts:
            # Sort: CYPs first
            CYP_FIRST = ["CYP3A4","CYP2D6","CYP2C9","CYP1A2","CYP2C19"]
            def _sort_key(k):
                return (0, CYP_FIRST.index(k)) if k in CYP_FIRST else (1, k)
            sorted_isos = sorted(iso_counts.keys(), key=_sort_key)

            legend_html = " ".join(
                f'<span class="pipe-badge isoform-{iso}" '
                f'title="{iso_counts[iso]} matched atoms">{iso}</span>'
                for iso in sorted_isos
            )
            st.markdown(
                f'<div style="margin-top:0.7rem;padding:0.5rem 0.6rem;'
                f'background:#F5F6F7;border:1px solid #D8D8D8;'
                f'border-radius:8px;display:flex;align-items:center;flex-wrap:wrap;gap:4px;">'
                f'<span style="font-size:0.68rem;color:#6B7078;'
                f'font-family:var(--font-display);font-weight:700;'
                f'letter-spacing:0.08em;text-transform:uppercase;'
                f'margin-right:4px;white-space:nowrap;">Enzyme targets</span>'
                f'{legend_html}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="svg-frame">
                <div class="empty-state">
                    <div class="empty-icon">⊘</div>
                    <div class="empty-text">
                        No soft-spot atoms identified.<br>
                        SMARTCyp found no matching P450 patterns for this structure.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_results_table(
    df: pd.DataFrame,
    parent: dict[str, Any],
    stats: dict[str, Any],
    soft_spot_summary: dict[str, Any] | None = None,
    compound_name: str = "",
) -> None:
    st.markdown('<div class="section-header">LC-MS Screening Target List</div>',
                unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        phase_opts   = ["All Phases"] + sorted(df["Phase"].unique().tolist())
        phase_filter = st.selectbox("Phase", phase_opts, label_visibility="collapsed")
    with col_f2:
        status_opts   = ["All Confidence Levels"] + sorted(df["Consensus Status"].unique().tolist())
        status_filter = st.selectbox("Status", status_opts, label_visibility="collapsed")
    with col_f3:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05,
                              label_visibility="collapsed",
                              help="Filter by minimum ensemble score")

    filtered = df.copy()
    if phase_filter != "All Phases":
        filtered = filtered[filtered["Phase"] == int(phase_filter)]
    if status_filter != "All Confidence Levels":
        filtered = filtered[filtered["Consensus Status"] == status_filter]
    filtered = filtered[filtered["Ensemble Score"] >= min_score]

    st.caption(f"Showing **{len(filtered)}** of **{len(df)}** metabolites · sorted by |Δm/z|")

    display_cols = [
        "Rank", "Structure (SMILES)", "Transformation", "Enzyme",
        "Neutral Mass (Da)", "Δ m/z", "[M+H]⁺ m/z", "[M-H]⁻ m/z",
        "Phase", "Ensemble Score", "Consensus Status",
    ]
    display_df = filtered[display_cols].copy()
    display_df["Neutral Mass (Da)"] = display_df["Neutral Mass (Da)"].map(lambda v: f"{v:.4f}")
    display_df["[M+H]⁺ m/z"]       = display_df["[M+H]⁺ m/z"].map(lambda v: f"{v:.4f}")
    display_df["[M-H]⁻ m/z"]       = display_df["[M-H]⁻ m/z"].map(lambda v: f"{v:.4f}")
    display_df["Ensemble Score"]    = display_df["Ensemble Score"].map(lambda v: f"{v:.4f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(420, 56 + 35 * len(display_df)),
        column_config={
            "Rank":               st.column_config.NumberColumn("Rank", width="small"),
            "Structure (SMILES)": st.column_config.TextColumn("SMILES", width="large"),
            "Transformation":     st.column_config.TextColumn(
                                      "Transformation",
                                      width="medium",
                                      help="Metabolic transformation type (hydroxylation, glucuronidation, etc.)"
                                  ),
            "Enzyme":             st.column_config.TextColumn(
                                      "Enzyme",
                                      width="small",
                                      help="Primary enzyme class responsible for this transformation"
                                  ),
            "Neutral Mass (Da)":  st.column_config.TextColumn("Neutral Mass (Da)", width="medium"),
            "Δ m/z":              st.column_config.TextColumn("Δ m/z", width="small"),
            "[M+H]⁺ m/z":        st.column_config.TextColumn("[M+H]⁺", width="medium"),
            "[M-H]⁻ m/z":        st.column_config.TextColumn("[M-H]⁻", width="medium"),
            "Ensemble Score":     st.column_config.TextColumn("Score", width="small"),
            "Consensus Status":   st.column_config.TextColumn("Consensus", width="medium"),
        },
    )

    st.markdown("")
    pdf_top_n = st.number_input(
        "Metabolites in PDF report",
        min_value=5, max_value=30, value=12, step=1,
        help="Number of top-ranked metabolites to include in the PDF report (10–15 recommended).",
    )
    dlp, dl1, dl2, dl3 = st.columns([2, 2, 2, 3])
    with dlp:
        try:
            _pa = parent.get("adducts", {})
            _pm = pd.DataFrame(
                [
                    ["Parent SMILES",             parent.get("smiles", "")],
                    ["Molecular Formula",         parent.get("molecular_formula", "")],
                    ["Neutral Monoisotopic Mass", parent.get("neutral_mass", "")],
                    ["[M+H]+ m/z",                _pa.get("mplus_h", "")],
                    ["[M-H]- m/z",                _pa.get("mminus_h", "")],
                ],
                columns=["Parameter", "Value"],
            )
            _atom_scores_json = json.dumps((soft_spot_summary or {}).get("atom_scores", []))
            with st.spinner("Building PDF report…"):
                _pdf = _build_pdf_bytes(df, _pm, compound_name, int(pdf_top_n), _atom_scores_json)
            st.download_button(
                label="⬇  PDF report",
                data=_pdf,
                file_name="hrms_predict_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"PDF unavailable: {exc}")
    with dl1:
        st.download_button(
            label="⬇  Export .xlsx",
            data=_build_excel_bytes(filtered, parent),
            file_name="hrms_metabolites.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="⬇  Export .csv",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="hrms_metabolites.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl3:
        st.caption(
            f"SyGMa: {stats.get('sygma_count',0)}  ·  "
            f"BioTransformer: {stats.get('biotransformer_count',0)}  ·  "
            f"DL: {stats.get('dl_count',0)}  ·  "
            f"SMARTCyp: {stats.get('smartcyp_count',0)}"
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict[str, Any]:
    """Render sidebar using plain HTML sections — no expanders, no arrow overlap."""

    def _section(title: str) -> None:
        """Render a clean section header inside the sidebar."""
        st.sidebar.markdown(
            f'<div style="font-family:\'Helvetica Neue\',sans-serif;'
            f'font-size:0.68rem;font-weight:700;letter-spacing:0.11em;'
            f'text-transform:uppercase;color:#6B7078;'
            f'margin:1rem 0 0.5rem;padding-bottom:0.35rem;'
            f'border-bottom:1.5px solid #E6EAF2;">{title}</div>',
            unsafe_allow_html=True,
        )

    # ── Header ─────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="font-family:\'Helvetica Neue\',sans-serif;'
        'font-size:0.68rem;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#6B7078;'
        'padding-bottom:0.5rem;border-bottom:1.5px solid #E6EAF2;"'
        '>Pipeline Configuration</div>',
        unsafe_allow_html=True,
    )

    # ── Active pipelines ───────────────────────────────────────────────
    _section("Active Pipelines")

    run_sygma = st.sidebar.checkbox(
        "SyGMa — Phase I/II rules",
        value=True,
        help="Rule-based metabolite generation via SMARTS transforms. Ready to use."
    )
    run_smartcyp = st.sidebar.checkbox(
        "SMARTCyp — P450 + non-CYP soft spots",
        value=True,
        help="69 SMARTS rules across 11 enzyme classes. Ready to use."
    )

    # ── Unavailable pipelines (greyed out, informational only) ────────
    st.sidebar.markdown(
        """
        <div style="margin-top:0.8rem;padding-top:0.75rem;
                    border-top:1px solid #D8D8D8;">
          <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;
                      text-transform:uppercase;color:#CFD0D2;margin-bottom:0.55rem;">
            Optional — requires setup
          </div>
          <div style="background:#F5F6F7;border:1px solid #EEEFF1;
                      border-radius:8px;padding:0.55rem 0.7rem;margin-bottom:0.4rem;">
            <div style="font-size:0.82rem;color:#CFD0D2;
                        text-decoration:line-through;margin-bottom:0.18rem;">
              BioTransformer
            </div>
            <div style="font-size:0.72rem;color:#9AA0A6;">
              Needs BioTransformer3.0.jar — see README
            </div>
          </div>
          <div style="background:#F5F6F7;border:1px solid #EEEFF1;
                      border-radius:8px;padding:0.55rem 0.7rem;">
            <div style="font-size:0.82rem;color:#CFD0D2;
                        text-decoration:line-through;margin-bottom:0.18rem;">
              DL Transformer (MetaTrans)
            </div>
            <div style="font-size:0.72rem;color:#9AA0A6;">
              Needs HuggingFace model weights — see README
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Force unavailable pipelines off
    run_biotransformer = False
    run_dl             = False

    # ── SyGMa settings ─────────────────────────────────────────────────
    _section("SyGMa Settings")

    p1_cycles = st.sidebar.slider(
        "Phase I cycles", 1, 3, 1,
        help="Number of successive Phase I rule applications"
    )
    p2_cycles = st.sidebar.slider(
        "Phase II cycles", 1, 3, 1,
        help="Number of successive Phase II rule applications"
    )

    # ── SMARTCyp settings ──────────────────────────────────────────────
    _section("SMARTCyp Settings")

    ea_cutoff = st.sidebar.slider(
        "Max Ea cutoff (kcal/mol)", 70.0, 110.0, 95.0, 1.0,
        help="Lower = stricter — fewer but higher-confidence soft-spot matches"
    )

    # ── Visualisation settings ─────────────────────────────────────────
    _section("Visualisation")

    scheme_label = st.sidebar.selectbox(
        "Soft-spot colour scheme",
        list(COLOUR_SCHEMES.keys()),
        index=0
    )
    alpha_max = st.sidebar.slider(
        "Highlight opacity", 0.2, 1.0, 0.80, 0.05,
        help="Controls glow intensity for high-risk atoms"
    )

    # ── CDD Vault connection (in-app login; held only in session memory) ──
    st.sidebar.markdown('<div class="section-header">CDD Vault</div>',
                        unsafe_allow_html=True)
    with st.sidebar.expander("🔗  Connect to CDD", expanded=False):
        st.text_input("Vault ID", key="cdd_vault_id",
                      help="Numeric Vault ID from your CDD Vault URL.")
        st.text_input("API token", key="cdd_token", type="password",
                      help="Generated in CDD account settings. Held only in this "
                           "session in memory; never written to disk or logged.")
        if st.session_state.get("cdd_vault_id") and st.session_state.get("cdd_token"):
            st.caption("✓ Connected for this session — GEN-ID lookup enabled.")
        else:
            st.caption("Enter Vault ID + API token to enable GEN-ID lookup.")

    # Defaults for unused pipelines
    bt_type   = "allHuman"
    dl_device = "cpu"

    return {
        "run_sygma":           run_sygma,
        "run_biotransformer":  run_biotransformer,
        "run_dl":              run_dl,
        "run_smartcyp":        run_smartcyp,
        "sygma_phase1_cycles": p1_cycles,
        "sygma_phase2_cycles": p2_cycles,
        "biotransformer_type": bt_type,
        "dl_device":           dl_device,
        "smartcyp_ea_cutoff":  ea_cutoff,
        "colour_scheme":       COLOUR_SCHEMES[scheme_label],
        "alpha_max":           alpha_max,
    }


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()

    backend_online = _check_backend_health()
    _render_header(backend_online)
    config = _render_sidebar()

    # ── Structure input ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Structure Input</div>',
                unsafe_allow_html=True)

    # Diclofenac — reliable SyGMa substrate, well-covered aromatic + N-H rules
    DEFAULT_SMILES = "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl"

    tab1, tab2, tab3 = st.tabs(
        ["✏️  Draw Structure", "⌨️  Paste SMILES", "🔗  Fetch from CDD"])

    with tab1:
        st.caption("Draw your parent compound. SMILES is captured automatically on each edit.")
        ketcher_smiles = st_ketcher(DEFAULT_SMILES, height=440)
        smiles_from_ketcher = ketcher_smiles if ketcher_smiles else DEFAULT_SMILES

    with tab2:
        smiles_text = st.text_input(
            "SMILES string",
            value=DEFAULT_SMILES,
            placeholder="e.g. O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
            label_visibility="collapsed",
        )

    with tab3:
        st.caption("Pull the parent structure straight from CDD Vault by GEN-ID.")
        _creds_set = bool(st.session_state.get("cdd_vault_id")
                          and st.session_state.get("cdd_token"))
        if not _creds_set:
            st.info("Connect to CDD in the sidebar (Vault ID + API token) to enable lookup.")
        _gc, _bc = st.columns([3, 1])
        with _gc:
            gen_id = st.text_input("GEN-ID", key="cdd_gen_id",
                                   placeholder="e.g. GEN-0016770",
                                   label_visibility="collapsed")
        with _bc:
            _fetch = st.button("Fetch", use_container_width=True, disabled=not _creds_set)
        if _fetch and gen_id.strip():
            try:
                with st.spinner(f"Looking up {gen_id.strip()} in CDD Vault…"):
                    _smi, _matched = fetch_smiles_by_name(
                        gen_id, st.session_state["cdd_vault_id"],
                        st.session_state["cdd_token"])
                st.session_state["cdd_smiles"] = _smi
                st.session_state["cdd_matched_name"] = _matched
                st.success(f"✓ Retrieved {_matched} from CDD.")
            except CDDError as _e:
                st.error(str(_e))
            except Exception as _e:  # noqa: BLE001
                st.error(f"CDD lookup failed: {_e}")
        if st.session_state.get("cdd_smiles"):
            st.markdown(
                f'**Fetched ({st.session_state.get("cdd_matched_name","")}):** '
                f'`{st.session_state["cdd_smiles"]}`')
            if st.button("Clear CDD structure"):
                st.session_state.pop("cdd_smiles", None)
                st.session_state.pop("cdd_matched_name", None)

    # Resolve the active SMILES: a freshly typed string wins; else a
    # CDD-fetched structure; else the Ketcher drawing; else the text box.
    _cdd_smiles = st.session_state.get("cdd_smiles", "").strip()
    _typed = smiles_text.strip()
    if _typed and _typed not in (DEFAULT_SMILES, _cdd_smiles):
        active_smiles = _typed
    elif _cdd_smiles:
        active_smiles = _cdd_smiles
    elif smiles_from_ketcher:
        active_smiles = smiles_from_ketcher.strip()
    else:
        active_smiles = _typed

    # ── Run button ─────────────────────────────────────────────────────
    st.markdown("")
    btn_col, info_col = st.columns([2, 5])

    with btn_col:
        run_btn = st.button(
            "⚗  Run Prediction",
            use_container_width=True,
            disabled=not backend_online,
            help=f"Requires the FastAPI backend on port {_BACKEND_PORT}" if not backend_online
                 else "Run all enabled prediction pipelines",
        )

    with info_col:
        if not backend_online:
            st.warning(
                f"Backend is offline. Start it with: "
                f"`uvicorn app.main:app --reload --port {_BACKEND_PORT}`",
                icon="⚠️",
            )
            if globals().get("_IP_ERR"):
                with st.expander("In-process engine error (diagnostic)", expanded=True):
                    st.code(globals()["_IP_ERR"])
        elif active_smiles:
            st.markdown(
                f'<div style="padding-top:0.5rem;">'
                f'<span style="font-family:var(--font-mono);font-size:0.78rem;'
                f'color:var(--text-muted);">Active SMILES → </span>'
                f'<code style="font-size:0.78rem;">{active_smiles[:80]}'
                f'{"…" if len(active_smiles) > 80 else ""}</code></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr style="margin:1.2rem 0;">', unsafe_allow_html=True)

    # ── Results dashboard ──────────────────────────────────────────────
    if run_btn or "last_result" in st.session_state:

        if run_btn:
            if not active_smiles:
                st.error("Please draw or paste a SMILES string before running.")
                st.stop()

            payload = {
                "smiles":               active_smiles,
                "run_sygma":            config["run_sygma"],
                "run_biotransformer":   config["run_biotransformer"],
                "run_dl":               config["run_dl"],
                "run_smartcyp":         config["run_smartcyp"],
                "sygma_phase1_cycles":  config["sygma_phase1_cycles"],
                "sygma_phase2_cycles":  config["sygma_phase2_cycles"],
                "biotransformer_type":  config["biotransformer_type"],
                "dl_device":            config["dl_device"],
                "smartcyp_ea_cutoff":   config["smartcyp_ea_cutoff"],
            }

            with st.spinner("Running prediction pipelines — this may take 30–120 s…"):
                try:
                    resp = _http_post(PREDICT_URL, json=payload, timeout=300)
                except Exception as exc:
                    st.error(f"Network error: {exc}")
                    st.stop()
                if resp.status_code != 200:
                    try:
                        detail = resp.json().get("detail", resp.text[:300])
                    except Exception:
                        detail = resp.text[:300]
                    st.error(f"Prediction failed (HTTP {resp.status_code}): {detail}")
                    st.stop()
                result = resp.json()
                st.session_state["last_result"] = result
                st.session_state["last_smiles"]  = active_smiles
                st.session_state["last_config"]  = config

        result      = st.session_state["last_result"]
        used_smiles = st.session_state.get("last_smiles", active_smiles)
        used_config = st.session_state.get("last_config", config)

        parent      = result.get("parent", {})
        metabolites = result.get("metabolites", [])
        stats       = result.get("pipeline_stats", {})
        soft_sum    = result.get("soft_spot_summary", {})

        if not metabolites:
            st.info(
                "No metabolites were generated. "
                "Try enabling more pipelines or relaxing the SMARTCyp Ea cutoff.",
                icon="ℹ️",
            )
            st.stop()

        df = _build_display_df(metabolites)

        # ── 3-column dashboard ─────────────────────────────────────────
        col_left, col_mid, col_right = st.columns([1.4, 2.0, 2.6], gap="large")

        with col_left:
            st.markdown('<div class="section-header">Parent Compound Metrics</div>',
                        unsafe_allow_html=True)
            _render_kpi_cards(parent, stats, metabolites)

            top_atoms = soft_sum.get("top_atoms", [])[:8]
            if top_atoms:
                st.markdown(
                    '<div class="section-header" style="margin-top:1rem;">'
                    'Top Vulnerable Atoms</div>',
                    unsafe_allow_html=True,
                )
                atom_html = " ".join(
                    f'<span style="display:inline-block;background:#FEF3C7;'
                    f'color:#92400E;border:1px solid #FDE68A;border-radius:4px;'
                    f'font-family:var(--font-mono);font-size:0.72rem;'
                    f'padding:1px 8px;margin:2px;">#{a}</span>'
                    for a in top_atoms
                )
                st.markdown(atom_html, unsafe_allow_html=True)

                top_rules = soft_sum.get("top_rules", [])[:3]
                if top_rules:
                    st.markdown(
                        '<div style="margin-top:0.6rem;font-size:0.72rem;'
                        'color:var(--text-muted);font-family:var(--font-mono);">'
                        + "<br>".join(f"· {r}" for r in top_rules)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

        with col_mid:
            _render_svg_panel(
                smiles    = used_smiles,
                metabolites = metabolites,
                scheme    = used_config.get("colour_scheme", "risk"),
                alpha_max = used_config.get("alpha_max", 0.70),
                soft_spot_summary = result.get("soft_spot_summary", {}),
            )

        with col_right:
            _render_results_table(
                df, parent, stats,
                soft_spot_summary=result.get("soft_spot_summary", {}),
                compound_name=st.session_state.get("cdd_matched_name", ""),
            )

    else:
        st.markdown(
            """
            <div class="empty-state" style="padding:4rem 2rem;">
                <div class="empty-icon">⚗️</div>
                <div class="empty-text">
                    <strong style="color:var(--text-secondary);
                                   font-family:var(--font-display);
                                   font-size:1.05rem;">Ready for analysis</strong>
                    <br><br>
                    Draw or paste a parent compound SMILES above,<br>
                    configure your pipeline options in the sidebar,<br>
                    then click <strong>Run Prediction</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
