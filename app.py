"""
app.py — Streamlit frontend for First Doctor Disease Detection Pipeline

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="First Doctor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a cleaner look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #5f6368;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1a73e8;
        margin-bottom: 0.5rem;
    }
    .metric-card h4 {
        margin: 0 0 0.3rem 0;
        color: #202124;
    }
    .metric-card p {
        margin: 0;
        color: #5f6368;
        font-size: 0.9rem;
    }
    /* Escalation badges */
    .badge-emergency { color: #d93025; font-weight: 700; }
    .badge-high      { color: #e37400; font-weight: 700; }
    .badge-urgent    { color: #e37400; font-weight: 600; }
    .badge-moderate  { color: #1a73e8; font-weight: 600; }
    /* Disclaimer */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — About the Model
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/stethoscope.png",
        width=64,
    )
    st.markdown("## About the Model")
    st.markdown(
        """
        **First Doctor** is a student-built AI triage and preliminary
        diagnostics system.

        **Pipeline:**
        1. 🔍 **Symptom Extraction** — spaCy NLP entity recognition
        2. 🧠 **Disease Prediction** — PyTorch neural network
        3. 🧪 **Test Mapping** — Fuzzy matching to lab test bundles
        4. 📄 **PDF Prescription** — Downloadable report

        **Model Architecture:**
        - 2-layer feed-forward classifier
        - 382 binary symptom inputs
        - Trained on 246k+ patient records

        **Tech Stack:**
        `spaCy` · `PyTorch` · `scikit-learn` · `thefuzz` · `fpdf` · `Streamlit`
        """
    )
    st.divider()
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#999;">
        ⚠️ This is a student project for educational purposes only.<br>
        Results are <b>NOT</b> medical diagnoses.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load the DiseaseDetector (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model & dataset …")
def load_detector():
    from disease_engine import DiseaseDetector

    return DiseaseDetector()


try:
    detector = load_detector()
    model_ready = True
except Exception as exc:
    model_ready = False
    model_error = str(exc)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🩺 First Doctor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">AI Triage &amp; Preliminary Diagnostics</p>',
    unsafe_allow_html=True,
)

if not model_ready:
    st.error(
        f"**Model failed to load.** Check that `model/database/dataset.csv` "
        f"is the real CSV (not a Git LFS pointer) and that all dependencies "
        f"are installed.\n\n```\n{model_error}\n```"
    )
    st.stop()

# ---- Input area ----
symptom_text = st.text_area(
    "Describe your symptoms",
    placeholder="e.g. I have a high fever and a persistent cough for 3 days …",
    height=120,
)

diagnose_clicked = st.button("🔍  Diagnose", type="primary", use_container_width=True)

# ---- Run pipeline ----
if diagnose_clicked:
    if not symptom_text.strip():
        st.warning("Please describe your symptoms before clicking Diagnose.")
        st.stop()

    with st.spinner("Analysing symptoms …"):
        result = detector.run(symptom_text.strip())

    # Store in session state so the PDF button works after re-render
    st.session_state["last_result"] = result
    st.session_state["last_input"] = symptom_text.strip()

# ---- Display results ----
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    symptoms = result["extracted_symptoms"]
    predictions = result["predictions"]

    st.divider()

    # -- Extracted symptoms chips --
    st.markdown("### 🔎 Extracted Symptoms")
    if symptoms:
        cols = st.columns(min(len(symptoms), 6))
        for i, sym in enumerate(symptoms):
            cols[i % len(cols)].markdown(
                f"<span style='background:#e8f0fe; color:#1a73e8; "
                f"padding:4px 10px; border-radius:12px; font-size:0.85rem;'>"
                f"{sym.replace('_', ' ').title()}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No specific symptoms detected. Try being more descriptive.")

    st.markdown("### 📋 Probable Diseases")

    # -- Accordion for each prediction --
    for i, pred in enumerate(predictions, 1):
        disease = pred["disease"]
        prob = pred["probability"] * 100
        known = pred.get("known_symptoms", [])
        rec = pred.get("recommended_tests", {})
        tests = rec.get("tests", [])
        esc = rec.get("escalation")

        # Build the expander label
        esc_badge = ""
        if esc:
            priority = esc["priority"]
            css_class = {
                "EMERGENCY": "badge-emergency",
                "HIGH": "badge-high",
                "URGENT": "badge-urgent",
            }.get(priority, "badge-moderate")
            esc_badge = f"  ⚠️ {priority}"

        with st.expander(
            f"**#{i}  {disease}**  —  {prob:.1f}%{esc_badge}", expanded=(i == 1)
        ):
            # Probability bar
            st.progress(min(pred["probability"], 1.0))

            # Known symptoms
            if known:
                st.markdown("**Associated Symptoms from Dataset:**")
                symptom_text_display = ", ".join(
                    s.replace("_", " ").title() for s in known[:20]
                )
                if len(known) > 20:
                    symptom_text_display += f" … and {len(known) - 20} more"
                st.markdown(f"_{symptom_text_display}_")

            # Recommended tests
            if tests:
                st.markdown("**Recommended Lab Tests:**")
                for test in tests:
                    st.markdown(f"- ✅ {test}")

            # Escalation alert
            if esc:
                priority = esc["priority"]
                if priority == "EMERGENCY":
                    st.error(f"🚨 **{priority}:** {esc['notes']}")
                elif priority in ("HIGH", "URGENT"):
                    st.warning(f"⚠️ **{priority}:** {esc['notes']}")
                else:
                    st.info(f"ℹ️ **{priority}:** {esc['notes']}")

    # -- PDF Download --
    st.divider()
    st.markdown("### 📄 Prescription PDF")

    pdf_path = detector.generate_pdf(
        result,
        patient_name=st.session_state.get("patient_name", "Patient"),
    )
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    col1, col2 = st.columns([3, 1])
    with col1:
        patient_name = st.text_input(
            "Patient name (for the PDF)",
            value=st.session_state.get("patient_name", ""),
            placeholder="Enter patient name …",
            key="patient_name_input",
        )
        if patient_name != st.session_state.get("patient_name", ""):
            st.session_state["patient_name"] = patient_name
            # Regenerate PDF with updated name
            pdf_path = detector.generate_pdf(result, patient_name=patient_name)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

    with col2:
        st.download_button(
            label="⬇️  Download Prescription PDF",
            data=pdf_bytes,
            file_name=f"first_doctor_prescription.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    # -- Disclaimer --
    st.markdown(
        """
        <div class="disclaimer">
            <strong>⚠️ DISCLAIMER:</strong> This is an AI-generated preliminary
            report from a student project. It is <strong>NOT</strong> a medical
            diagnosis. Please consult a certified medical professional before
            taking any action based on these results.
        </div>
        """,
        unsafe_allow_html=True,
    )
