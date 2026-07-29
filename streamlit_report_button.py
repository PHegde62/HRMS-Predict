# ── Add this block to your app.py (Streamlit frontend) ───────────────────────
# Place it AFTER the section that shows the metabolites table / soft-spot SVG
# and BEFORE the final st.stop() or end of the page.
#
# This adds a "Download PDF Report" button that generates and downloads
# the report directly from the Streamlit UI.

import streamlit as st
import tempfile, os, subprocess, sys

def add_report_download_button(xlsx_bytes, compound_name, top_n=12):
    """
    Call this after your results section with the xlsx bytes from your
    existing Excel export (the same bytes you use for the xlsx download button).

    Example usage in your app.py:
        if results:
            xlsx_bytes = generate_excel_output(results)   # your existing function
            add_report_download_button(xlsx_bytes, compound_name=smiles_input[:20])
    """
    st.divider()
    st.subheader("📄 Export Report")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        top_n = st.selectbox("Metabolites to include", [10, 12, 15], index=1)

    with col2:
        compound_label = st.text_input("Compound name (for report title)",
                                       value=compound_name or "Compound")

    with col3:
        st.write("")  # spacing

    if st.button("🖨️ Generate PDF Report", type="primary"):
        with st.spinner("Generating report (this takes ~5 seconds)..."):
            try:
                # Write xlsx to temp file
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
                    f.write(xlsx_bytes)
                    xlsx_tmp = f.name

                # Output PDF path
                pdf_tmp = xlsx_tmp.replace(".xlsx", "_report.pdf")

                # Call the report generator script
                script_path = os.path.join(os.path.dirname(__file__),
                                           "generate_report.py")
                result = subprocess.run(
                    [sys.executable, script_path,
                     xlsx_tmp,
                     "--compound", compound_label,
                     "--top", str(top_n),
                     "--out", pdf_tmp],
                    capture_output=True, text=True
                )

                if result.returncode != 0:
                    st.error(f"Report generation failed:\n{result.stderr}")
                    return

                # Read PDF and offer download
                with open(pdf_tmp, "rb") as f:
                    pdf_bytes = f.read()

                safe_name = compound_label.replace(" ", "_").lower()
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{safe_name}_hrms_predict_report.pdf",
                    mime="application/pdf",
                )
                st.success(f"Report generated: {len(pdf_bytes)//1024} KB, "
                           f"{top_n} metabolites, {compound_label}")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Clean up temp files
                for f in [xlsx_tmp, pdf_tmp]:
                    try: os.unlink(f)
                    except: pass


# ── Alternative: generate report directly from session state ─────────────────
# If your app stores predictions in session state rather than xlsx bytes,
# use this version instead:

def add_report_button_from_smiles(smiles, compound_name, predictions_df, top_n=12):
    """
    Generate report directly from a DataFrame of predictions (no xlsx needed).
    predictions_df must have the same columns as the Metabolites sheet.
    """
    import io
    import pandas as pd

    if st.button("🖨️ Generate PDF Report", key="pdf_report_btn", type="primary"):
        with st.spinner("Generating report..."):
            try:
                # Build xlsx in memory
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    predictions_df.to_excel(writer, sheet_name="Metabolites", index=False)
                    parent_metrics = pd.DataFrame({
                        "Parameter": ["Parent SMILES", "Molecular Formula",
                                      "Neutral Monoisotopic Mass"],
                        "Value": [smiles, "", ""]
                    })
                    parent_metrics.to_excel(writer, sheet_name="Parent Metrics", index=False)
                buf.seek(0)

                # Generate PDF
                add_report_download_button(buf.read(), compound_name, top_n)

            except Exception as e:
                st.error(f"Report error: {e}")
