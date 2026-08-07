"""
HRMS-Predict Report Generator
==============================
Generates a professional PDF report from an HRMS-Predict Excel output file.

Usage:
    python generate_report.py diclofenac.xlsx
    python generate_report.py diclofenac.xlsx --top 15 --out diclofenac_report.pdf
    python generate_report.py diclofenac.xlsx --compound "Diclofenac" --top 10

Requires:
    pip install reportlab rdkit pandas openpyxl

Place this file in your hrms-predict folder and run from the conda environment:
    conda activate hrms-predictor
    python generate_report.py your_output.xlsx
"""

import argparse, ast, io, os, sys
from datetime import datetime
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdfcanvas

# ── Enzyme colour map (matches the UI soft-spot colours) ──────────────────────
ENZYME_COLOURS = {
    "CYP":        (0.93, 0.62, 0.13),   # amber
    "CYP1A2":     (0.93, 0.62, 0.13),
    "CYP2C9":     (0.93, 0.62, 0.13),
    "CYP2C19":    (0.93, 0.62, 0.13),
    "CYP2D6":     (0.93, 0.62, 0.13),
    "CYP3A4":     (0.93, 0.62, 0.13),
    "UGT":        (0.11, 0.62, 0.46),   # teal
    "SULT":       (0.33, 0.55, 0.82),   # blue
    "NAT":        (0.58, 0.40, 0.74),   # purple
    "COMT":       (0.58, 0.40, 0.74),
    "COMT / TPMT":(0.58, 0.40, 0.74),
    "FMO":        (0.85, 0.37, 0.22),   # coral
    "FMO3":       (0.85, 0.37, 0.22),
    "AO":         (0.85, 0.37, 0.22),
    "MAO":        (0.62, 0.18, 0.18),   # dark red
    "GST":        (0.20, 0.63, 0.17),   # green
    "Spontaneous":(0.55, 0.55, 0.55),   # grey
    "DEFAULT":    (0.55, 0.55, 0.55),
}

ENZYME_HEX = {
    "CYP":        "#EE9E21",
    "CYP1A2":     "#EE9E21",
    "CYP2C9":     "#EE9E21",
    "CYP2C19":    "#EE9E21",
    "CYP2D6":     "#EE9E21",
    "CYP3A4":     "#EE9E21",
    "UGT":        "#1D9E75",
    "SULT":       "#5495D0",
    "NAT":        "#9466BD",
    "COMT":       "#9466BD",
    "COMT / TPMT":"#9466BD",
    "FMO":        "#D85E38",
    "FMO3":       "#D85E38",
    "AO":         "#D85E38",
    "MAO":        "#9E2E2E",
    "GST":        "#33A12B",
    "Spontaneous":"#8C8C8C",
    "DEFAULT":    "#8C8C8C",
}

def enzyme_rgb(enzyme_str):
    """Return RDKit-style (r,g,b) 0-1 float tuple for an enzyme string."""
    if not enzyme_str or str(enzyme_str).lower() in ("nan", "none", ""):
        return ENZYME_COLOURS["DEFAULT"]
    e = str(enzyme_str).strip()
    if e in ENZYME_COLOURS:
        return ENZYME_COLOURS[e]
    # Match prefix (e.g. "CYP2D6" → "CYP")
    for key in ENZYME_COLOURS:
        if e.upper().startswith(key.upper()):
            return ENZYME_COLOURS[key]
    return ENZYME_COLOURS["DEFAULT"]

def enzyme_hex(enzyme_str):
    """Return hex colour string for an enzyme."""
    if not enzyme_str or str(enzyme_str).lower() in ("nan", "none", ""):
        return ENZYME_HEX["DEFAULT"]
    e = str(enzyme_str).strip()
    if e in ENZYME_HEX:
        return ENZYME_HEX[e]
    for key in ENZYME_HEX:
        if e.upper().startswith(key.upper()):
            return ENZYME_HEX[key]
    return ENZYME_HEX["DEFAULT"]

def hex_to_rl(h):
    """Convert #RRGGBB to reportlab Color."""
    h = h.lstrip("#")
    return colors.Color(int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255)


def draw_molecule_png(smiles, highlight_atoms, enzyme, width=280, height=220):
    """
    Render a molecule to PNG bytes with enzyme-coloured atom highlights.
    Returns PNG bytes or None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)

        rgb = enzyme_rgb(enzyme)
        ha  = [int(a) for a in highlight_atoms if int(a) < mol.GetNumAtoms()]

        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.padding = 0.12
        opts.bondLineWidth = 1.8

        if ha:
            atom_cols = {i: rgb for i in ha}
            atom_rads = {i: 0.38 for i in ha}
            drawer.DrawMolecule(mol,
                highlightAtoms=ha,
                highlightAtomColors=atom_cols,
                highlightAtomRadii=atom_rads,
                highlightBonds=[])
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        print(f"  Warning: could not draw {smiles[:40]}: {e}")
        return None


def parse_soft_spot_atoms(val):
    """Parse Soft-Spot Atoms column — handles list strings like '[0, 1, 2]'."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if s in ("", "[]", "nan"):
        return []
    try:
        result = ast.literal_eval(s)
        return [int(x) for x in result]
    except:
        return []


def score_to_likelihood(score, phase):
    """
    Convert Ensemble Score to a qualitative likelihood label.
    Score is 0.0-1.0 from the tool; also use phase info.
    """
    try:
        s = float(score)
    except:
        return "Unknown"
    if s >= 0.7:    return "Very High (>70%)"
    if s >= 0.5:    return "High (50-70%)"
    if s >= 0.35:   return "Moderate (35-50%)"
    if s >= 0.20:   return "Low (20-35%)"
    return "Very Low (<20%)"


# ── ReportLab page template ───────────────────────────────────────────────────
# Genesis Therapeutics brand palette
NAVY   = colors.Color(0.0, 0.2745, 1.0)      # Genesis Blue  #0046FF (core)
TEAL   = colors.Color(0.0, 0.7843, 0.5216)   # AI green      #00C885 (accent, sparing)
LGRAY  = colors.Color(0.9608, 0.9647, 0.9686)  # neutral #F5F6F7
MGRAY  = colors.Color(0.8471, 0.8471, 0.8471)  # Integration #D8D8D8
WHITE  = colors.white
BLACK  = colors.black

W, H   = A4   # 595 x 842 pt
MARGIN = 18*mm

def header_footer(canvas, doc):
    """Draw header and footer on every page."""
    canvas.saveState()

    # ── Header bar ───────────────────────────────────────────────────────────
    canvas.setFillColor(NAVY)
    canvas.rect(0, H - 28*mm, W, 28*mm, fill=1, stroke=0)

    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(MARGIN, H - 14*mm, "HRMS\u00B7Predict")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(MARGIN, H - 20*mm, "In Silico Metabolite Prediction Report")

    # Tool info top-right
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(W - MARGIN, H - 14*mm,
        f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}")
    canvas.drawRightString(W - MARGIN, H - 20*mm,
        "github.com/PHegde62/HRMS-Predict")

    # ── Teal accent line under header ────────────────────────────────────────
    canvas.setFillColor(TEAL)
    canvas.rect(0, H - 29.5*mm, W, 1.5*mm, fill=1, stroke=0)

    # ── Footer ───────────────────────────────────────────────────────────────
    canvas.setFillColor(LGRAY)
    canvas.rect(0, 0, W, 12*mm, fill=1, stroke=0)
    canvas.setFillColor(MGRAY)
    canvas.rect(0, 12*mm, W, 0.5, fill=1, stroke=0)

    canvas.setFillColor(colors.Color(0.4,0.4,0.4))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(MARGIN, 4.5*mm,
        "For research use only. Predictions are computational estimates and do not substitute "
        "for experimental metabolic identification.")
    canvas.drawRightString(W - MARGIN, 4.5*mm, f"Page {doc.page}")

    canvas.restoreState()


def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["h1"] = ParagraphStyle("h1",
        fontName="Helvetica-Bold", fontSize=15, textColor=NAVY,
        spaceAfter=6, spaceBefore=14, leading=18)

    styles["h2"] = ParagraphStyle("h2",
        fontName="Helvetica-Bold", fontSize=11, textColor=NAVY,
        spaceAfter=4, spaceBefore=10, leading=14)

    styles["body"] = ParagraphStyle("body",
        fontName="Helvetica", fontSize=9, textColor=BLACK,
        spaceAfter=4, leading=13)

    styles["small"] = ParagraphStyle("small",
        fontName="Helvetica", fontSize=7.5, textColor=colors.Color(0.3,0.3,0.3),
        spaceAfter=2, leading=10)

    styles["mono"] = ParagraphStyle("mono",
        fontName="Courier", fontSize=7.5, textColor=colors.Color(0.2,0.2,0.2),
        spaceAfter=2, leading=10)

    styles["center"] = ParagraphStyle("center",
        fontName="Helvetica", fontSize=9, textColor=BLACK,
        alignment=TA_CENTER, spaceAfter=2, leading=12)

    styles["badge"] = ParagraphStyle("badge",
        fontName="Helvetica-Bold", fontSize=8, textColor=WHITE,
        alignment=TA_CENTER, leading=10)

    return styles


# ── Section builders ──────────────────────────────────────────────────────────

def section_cover(story, styles, compound_name, parent_smi, parent_metrics, n_predicted):
    """Page 1: cover with parent structure + key metrics."""

    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(compound_name, ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=22,
        textColor=NAVY, spaceAfter=2, alignment=TA_CENTER)))
    story.append(Paragraph("In Silico Metabolite Prediction Report",
        ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11,
        textColor=TEAL, spaceAfter=8, alignment=TA_CENTER)))

    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceAfter=8))

    # Parent molecule image (no highlighting)
    png = draw_molecule_png(parent_smi, [], "", width=320, height=260)
    if png:
        img = Image(io.BytesIO(png), width=8*cm, height=6.5*cm)
        img.hAlign = "CENTER"
        story.append(img)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(f"<b>Parent SMILES:</b> <font name='Courier' size='8'>{parent_smi}</font>",
        styles["small"]))
    story.append(Spacer(1, 4*mm))

    # Metrics grid
    metric_data = []
    for _, row in parent_metrics.iterrows():
        param = str(row.get("Parameter",""))
        val   = str(row.get("Value",""))
        if param and val and "smiles" not in param.lower():
            metric_data.append([param, val])
    metric_data.append(["Total metabolites predicted", str(n_predicted)])
    metric_data.append(["Report generated", datetime.now().strftime("%d %B %Y %H:%M")])

    t = Table([[Paragraph(f"<b>{r[0]}</b>", styles["body"]),
                Paragraph(r[1], styles["body"])]
               for r in metric_data],
              colWidths=[6.5*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), LGRAY),
        ("GRID",       (0,0), (-1,-1), 0.5, MGRAY),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, LGRAY]),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())


def section_softspot(story, styles, parent_smi, df_top, atom_scores=None):
    """
    Page 2: parent structure with ALL soft-spot atoms highlighted,
    colour-coded by enzyme class. Plus legend.
    """
    story.append(Paragraph("Soft-Spot Analysis", styles["h1"]))
    story.append(Paragraph(
        "Atoms highlighted below represent predicted metabolic soft-spots. "
        "Colour indicates the enzyme class responsible for each transformation. "
        "Atoms with multiple possible enzymes are shown in the colour of the "
        "highest-ranked enzyme.",
        styles["body"]))
    story.append(Spacer(1, 3*mm))

    # Build per-atom → enzyme mapping (highest rank wins)
    atom_enzyme = {}
    # Prefer the authoritative soft_spot_summary atom_scores (per-atom isoform);
    # fall back to per-metabolite Soft-Spot Atoms columns.
    if atom_scores:
        for entry in atom_scores:
            idx = entry.get("atom_idx")
            if idx is None:
                continue
            iso = (entry.get("isoform") or "").strip()
            atom_enzyme[int(idx)] = iso if iso else "DEFAULT"
    if not atom_enzyme:
        for _, row in df_top.iterrows():
            atoms = parse_soft_spot_atoms(row.get("Soft-Spot Atoms",""))
            enz   = str(row.get("Enzyme",""))
            for a in atoms:
                if a not in atom_enzyme:  # highest-ranked row comes first
                    atom_enzyme[a] = enz

    # Group atoms by enzyme for multi-colour rendering
    # RDKit supports per-atom colours in DrawMolecule
    try:
        mol = Chem.MolFromSmiles(str(parent_smi))
        if mol:
            AllChem.Compute2DCoords(mol)
            all_ha   = list(atom_enzyme.keys())
            atom_cols = {a: enzyme_rgb(e) for a, e in atom_enzyme.items()
                         if a < mol.GetNumAtoms()}
            atom_rads = {a: 0.40 for a in atom_cols}

            drawer = rdMolDraw2D.MolDraw2DCairo(420, 340)
            opts = drawer.drawOptions()
            opts.addAtomIndices = False
            opts.padding = 0.10
            opts.bondLineWidth = 2.0

            if atom_cols:
                drawer.DrawMolecule(mol,
                    highlightAtoms=list(atom_cols.keys()),
                    highlightAtomColors=atom_cols,
                    highlightAtomRadii=atom_rads,
                    highlightBonds=[])
            else:
                drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png = drawer.GetDrawingText()

            img = Image(io.BytesIO(png), width=11*cm, height=8.9*cm)
            img.hAlign = "CENTER"
            story.append(img)
    except Exception as e:
        story.append(Paragraph(f"(Structure rendering error: {e})", styles["small"]))

    story.append(Spacer(1, 4*mm))

    # Legend
    story.append(Paragraph("<b>Enzyme colour legend:</b>", styles["body"]))
    legend_enzymes = sorted(set(atom_enzyme.values()))
    if not legend_enzymes:
        legend_enzymes = ["CYP","UGT","SULT","NAT","FMO3","AO","GST","Spontaneous"]

    legend_rows = []
    row_buf = []
    for enz in legend_enzymes:
        hex_c = enzyme_hex(enz)
        swatch = Table([[""]],
            colWidths=[10], rowHeights=[10])
        swatch.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), hex_to_rl(hex_c)),
            ("GRID", (0,0), (0,0), 0, WHITE),
        ]))
        cell = Table([[swatch, Paragraph(enz, styles["small"])]],
            colWidths=[14, 45])
        cell.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),1),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),1),
                                   ("LEFTPADDING",(0,0),(-1,-1),2)]))
        row_buf.append(cell)
        if len(row_buf) == 4:
            legend_rows.append(row_buf[:])
            row_buf = []
    if row_buf:
        while len(row_buf) < 4:
            row_buf.append(Paragraph("", styles["small"]))
        legend_rows.append(row_buf)

    if legend_rows:
        legend_t = Table(legend_rows, colWidths=[65]*4)
        legend_t.setStyle(TableStyle([
            ("TOPPADDING",(0,0),(-1,-1),2),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ]))
        story.append(legend_t)

    story.append(Spacer(1, 3*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MGRAY))
    story.append(Spacer(1, 3*mm))


def section_metabolite_table(story, styles, df_top, parent_smi):
    """
    Main table: top 10-15 metabolites with structure image, enzyme badge,
    masses, transformation, probability.
    """
    story.append(Paragraph(f"Top {len(df_top)} Predicted Metabolites", styles["h1"]))
    story.append(Paragraph(
        "Metabolites ranked by ensemble score (highest confidence first). "
        "Structures show metabolic soft-spot atoms highlighted in enzyme colour. "
        "Probability reflects the ensemble score from the prediction pipeline "
        "(0 = not predicted; 1 = maximum confidence).",
        styles["body"]))
    story.append(Spacer(1, 3*mm))

    for idx, (_, row) in enumerate(df_top.iterrows()):
        rank   = int(row.get("Rank", idx+1))
        smi    = str(row.get("Structure (SMILES)", ""))
        txn    = str(row.get("Transformation", ""))
        enz    = str(row.get("Enzyme", ""))
        src    = str(row.get("Source Engine", ""))
        formula= str(row.get("Formula", ""))
        neutral= row.get("Neutral Mass (Da)", "")
        mhp    = row.get("[M+H]\u207a m/z", row.get("[M+H]+ m/z", ""))
        mhm    = row.get("[M-H]\u207b m/z", row.get("[M-H]- m/z", ""))
        delta  = row.get("\u0394 m/z", row.get("Δ m/z", ""))
        phase  = row.get("Phase", "")
        score  = row.get("Ensemble Score", 0)
        dl_conf= row.get("DL Confidence", 0)
        consensus= str(row.get("Consensus Status", ""))
        atoms  = parse_soft_spot_atoms(row.get("Soft-Spot Atoms",""))
        likelihood = score_to_likelihood(score, phase)

        # Format numbers
        def fmt(v, dp=4):
            try: return f"{float(v):.{dp}f}"
            except: return str(v)

        hex_c = enzyme_hex(enz)
        rl_c  = hex_to_rl(hex_c)
        phase_str = {0:"Phase 0 (parent)", 1:"Phase I", 2:"Phase II"}.get(
            int(phase) if str(phase).isdigit() else -1, str(phase))

        # Structure image with soft-spot colouring
        png = draw_molecule_png(smi, atoms, enz, width=260, height=210)
        if png:
            struct_img = Image(io.BytesIO(png), width=5.2*cm, height=4.2*cm)
        else:
            struct_img = Paragraph("(no structure)", styles["small"])

        # Enzyme badge
        badge_t = Table([[Paragraph(enz, ParagraphStyle(
            "badge2", fontName="Helvetica-Bold", fontSize=8.5,
            textColor=WHITE, alignment=TA_CENTER, leading=11))]],
            colWidths=[5.2*cm])
        badge_t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), rl_c),
            ("TOPPADDING",  (0,0),(0,0), 3),
            ("BOTTOMPADDING",(0,0),(0,0),3),
            ("LEFTPADDING", (0,0),(0,0), 4),
            ("RIGHTPADDING",(0,0),(0,0), 4),
        ]))

        # Left column: structure + badge + rank
        rank_para = Paragraph(
            f"<b>Rank #{rank}</b>",
            ParagraphStyle("rank", fontName="Helvetica-Bold", fontSize=9,
                textColor=NAVY, alignment=TA_CENTER))

        left_col = Table([
            [rank_para],
            [struct_img],
            [badge_t],
        ], colWidths=[5.5*cm])
        left_col.setStyle(TableStyle([
            ("ALIGN",  (0,0),(-1,-1),"CENTER"),
            ("VALIGN", (0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),2),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ]))

        # Right column: all data fields
        def kv(key, val, mono=False):
            font = "Courier" if mono else "Helvetica"
            return Paragraph(
                f"<b>{key}:</b> <font name='{font}' size='8'>{val}</font>",
                styles["body"])

        score_bar_pct = min(int(float(score)*100), 100) if score else 0

        right_items = [
            kv("Transformation", txn),
            kv("Formula", formula),
            kv("Neutral mass", f"{fmt(neutral)} Da"),
            kv("[M+H]\u207a", f"{fmt(mhp)} m/z"),
            kv("[M-H]\u207b", f"{fmt(mhm)} m/z"),
            kv("\u0394 m/z vs parent", f"{fmt(delta, 4)} Da"),
            kv("Phase", phase_str),
            kv("Source engine", src),
            kv("Consensus", consensus),
            Spacer(1, 2),
            Paragraph(f"<b>Ensemble score:</b> {fmt(score,3)}  "
                      f"<b>Likelihood:</b> {likelihood}",
                      styles["body"]),
            Paragraph(
                f"<font size='7' color='#888888'>SMILES: {smi[:80]}{'...' if len(smi)>80 else ''}</font>",
                styles["small"]),
        ]

        right_col = Table([[r] for r in right_items], colWidths=[10.8*cm])
        right_col.setStyle(TableStyle([
            ("TOPPADDING",   (0,0),(-1,-1),1),
            ("BOTTOMPADDING",(0,0),(-1,-1),1),
            ("LEFTPADDING",  (0,0),(-1,-1),0),
        ]))

        # Combined row
        row_bg = LGRAY if idx % 2 == 0 else WHITE
        card = Table([[left_col, right_col]],
            colWidths=[5.8*cm, 11.2*cm])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), row_bg),
            ("BOX",        (0,0),(-1,-1), 0.8, MGRAY),
            ("LINEAFTER",  (0,0),(0,-1),  0.5, MGRAY),
            ("VALIGN",     (0,0),(-1,-1), "TOP"),
            ("TOPPADDING", (0,0),(-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1), 6),
            ("RIGHTPADDING",(0,0),(-1,-1),6),
        ]))

        story.append(KeepTogether([card, Spacer(1, 3*mm)]))


def section_summary_table(story, styles, df_top):
    """
    Compact reference table at the end: all top metabolites in one table.
    """
    story.append(PageBreak())
    story.append(Paragraph("Summary Reference Table", styles["h1"]))
    story.append(Paragraph(
        "Compact reference table for all top predicted metabolites. "
        "Suitable for use as an LC-HRMS target list.",
        styles["body"]))
    story.append(Spacer(1, 3*mm))

    header = ["Rank", "Transformation", "Enzyme", "Formula",
              "Neutral\nMass (Da)", "[M+H]+\nm/z", "[M-H]-\nm/z",
              "Delta\nm/z", "Phase", "Score", "Likelihood"]

    def fmt(v, dp=4):
        try: return f"{float(v):.{dp}f}"
        except: return str(v)

    table_data = [header]
    for _, row in df_top.iterrows():
        enz   = str(row.get("Enzyme",""))
        score = row.get("Ensemble Score", 0)
        phase = row.get("Phase","")
        phase_s = {0:"Ph.0", 1:"Ph.I", 2:"Ph.II"}.get(
            int(phase) if str(phase).isdigit() else -1, str(phase))
        table_data.append([
            str(int(row.get("Rank", 0))),
            str(row.get("Transformation","")),
            enz,
            str(row.get("Formula","")),
            fmt(row.get("Neutral Mass (Da)","")),
            fmt(row.get("[M+H]\u207a m/z", row.get("[M+H]+ m/z",""))),
            fmt(row.get("[M-H]\u207b m/z", row.get("[M-H]- m/z",""))),
            fmt(row.get("\u0394 m/z", row.get("Δ m/z",""))),
            phase_s,
            fmt(score, 3),
            score_to_likelihood(score, phase).split(" ")[0],
        ])

    col_widths = [1.0*cm, 4.0*cm, 2.2*cm, 1.8*cm,
                  2.0*cm, 2.0*cm, 2.0*cm,
                  1.6*cm, 1.3*cm, 1.3*cm, 2.0*cm]

    t = Table(table_data, colWidths=col_widths, repeatRows=1)

    ts = TableStyle([
        # Header
        ("BACKGROUND",    (0,0),(-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,0),  7.5),
        ("ALIGN",         (0,0),(-1,0),  "CENTER"),
        ("TOPPADDING",    (0,0),(-1,0),  4),
        ("BOTTOMPADDING", (0,0),(-1,0),  4),
        # Body
        ("FONTSIZE",      (0,1),(-1,-1), 7.5),
        ("ALIGN",         (0,1),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,1),(-1,-1), 3),
        ("BOTTOMPADDING", (0,1),(-1,-1), 3),
        ("GRID",          (0,0),(-1,-1), 0.4, MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
    ])

    # Colour enzyme column cells
    for i, row_data in enumerate(table_data[1:], start=1):
        enz = row_data[2]
        hex_c = enzyme_hex(enz)
        ts.add("BACKGROUND", (2,i),(2,i), hex_to_rl(hex_c))
        ts.add("TEXTCOLOR",  (2,i),(2,i), WHITE)
        ts.add("FONTNAME",   (2,i),(2,i), "Helvetica-Bold")

    t.setStyle(ts)
    story.append(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_report(xlsx_path, compound_name=None, top_n=12, out_path=None):

    print(f"\nLoading: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name="Metabolites")
    try:
        pm = pd.read_excel(xlsx_path, sheet_name="Parent Metrics")
    except:
        pm = pd.DataFrame(columns=["Parameter","Value"])

    # Get parent SMILES
    parent_smi = ""
    pm_smiles = pm[pm["Parameter"].str.contains("SMILES", case=False, na=False)]
    if not pm_smiles.empty:
        parent_smi = str(pm_smiles.iloc[0]["Value"])

    # Compound name from file if not given
    if not compound_name:
        compound_name = os.path.splitext(os.path.basename(xlsx_path))[0].capitalize()

    # Sort by Rank (ascending), take top N
    df = df.sort_values("Rank").reset_index(drop=True)
    # Remove rank 1 if it's the parent (delta = 0.0 and same formula as parent)
    parent_formula = pm[pm["Parameter"].str.contains("Formula", case=False, na=False)]
    if not parent_formula.empty:
        pf = str(parent_formula.iloc[0]["Value"])
        parent_rows = df[(df["\u0394 m/z"].astype(float).abs() < 0.001) &
                         (df["Formula"] == pf)]
        if len(parent_rows) > 0:
            df = df.drop(parent_rows.index)

    df_top = df.head(top_n).reset_index(drop=True)
    print(f"Compound: {compound_name}  |  Top {len(df_top)} metabolites from {len(df)} total")

    # Output path
    if not out_path:
        base = os.path.splitext(xlsx_path)[0]
        out_path = f"{base}_report.pdf"

    # Build PDF
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        topMargin=32*mm,
        bottomMargin=18*mm,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        title=f"HRMS-Predict Report: {compound_name}",
        author="HRMS-Predict",
    )

    styles = build_styles()
    story  = []

    print("Building cover page...")
    section_cover(story, styles, compound_name, parent_smi, pm, len(df))

    print("Building soft-spot section...")
    section_softspot(story, styles, parent_smi, df_top)

    print("Building metabolite cards...")
    section_metabolite_table(story, styles, df_top, parent_smi)

    print("Building summary table...")
    section_summary_table(story, styles, df_top)

    print(f"Rendering PDF -> {out_path}")
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Done! Report saved to: {out_path}")
    return out_path


def build_report_from_frames(df, pm, compound_name="Predicted Compound",
                             top_n=12, atom_scores=None) -> bytes:
    """Build the PDF from in-memory frames (Streamlit app) and return PDF bytes."""
    df = df.copy()
    parent_smi = ""
    try:
        if "Parameter" in pm.columns:
            pm_s = pm[pm["Parameter"].astype(str).str.contains("SMILES", case=False, na=False)]
            if not pm_s.empty:
                parent_smi = str(pm_s.iloc[0]["Value"])
    except Exception:
        pass
    if "Rank" in df.columns:
        try:
            df = df.sort_values("Rank").reset_index(drop=True)
        except Exception:
            pass
    df_top = df.head(int(top_n)).reset_index(drop=True)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, topMargin=32*mm, bottomMargin=18*mm,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title=f"HRMS-Predict Report: {compound_name}", author="HRMS-Predict",
    )
    styles = build_styles()
    story = []
    section_cover(story, styles, compound_name, parent_smi, pm, len(df))
    section_softspot(story, styles, parent_smi, df_top, atom_scores=atom_scores)
    section_metabolite_table(story, styles, df_top, parent_smi)
    section_summary_table(story, styles, df_top)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a PDF report from an HRMS-Predict Excel output file.")
    parser.add_argument("xlsx", help="Path to the .xlsx output file from HRMS-Predict")
    parser.add_argument("--compound", "-c", default=None,
        help="Compound name for the report title (default: filename)")
    parser.add_argument("--top", "-t", type=int, default=12,
        help="Number of top metabolites to show (default: 12)")
    parser.add_argument("--out", "-o", default=None,
        help="Output PDF path (default: <xlsx_name>_report.pdf)")
    args = parser.parse_args()

    if not os.path.exists(args.xlsx):
        print(f"ERROR: File not found: {args.xlsx}", file=sys.stderr)
        sys.exit(1)

    generate_report(args.xlsx, compound_name=args.compound,
                    top_n=args.top, out_path=args.out)
