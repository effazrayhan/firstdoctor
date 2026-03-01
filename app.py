from __future__ import annotations

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="First Doctor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    
    /* Chat container */
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 120px;
    }
    
    /* Message bubbles */
    .user-message {
        background: #e8f0fe;
        padding: 0.8rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        color: #202124;
    }
    
    .assistant-message {
        background: #f1f3f4;
        padding: 0.8rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-right: 20%;
        color: #202124;
    }
    
    /* Fixed bottom input */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prescription_data" not in st.session_state:
    st.session_state.prescription_data = None

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

# Restart button at the top
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🔄 Restart", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.prescription_data = None
        st.rerun()

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-message">👤 {message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-message">🩺 {message["content"]}</div>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)

# Fixed bottom input area
st.markdown("---")
symptom_text = st.chat_input("Type your message or describe your symptoms...")

if symptom_text:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": symptom_text})
    
    # Check if this looks like a symptom description (simple heuristic)
    # You can make this more sophisticated
    needs_diagnosis = any(word in symptom_text.lower() for word in [
        "pain", "fever", "cough", "headache", "ache", "hurt", "sick", 
        "symptom", "feeling", "diagnose", "diagnosis"
    ]) or len(st.session_state.messages) >= 3
    
    if needs_diagnosis:
        with st.spinner("Analysing symptoms …"):
            # Combine all user messages for comprehensive analysis
            all_symptoms = " ".join([
                msg["content"] for msg in st.session_state.messages 
                if msg["role"] == "user"
            ])
            result = detector.run(all_symptoms)
            
            if result["extracted_symptoms"]:
                st.session_state.prescription_data = {
                    "result": result,
                    "input_text": all_symptoms
                }
                response = "I've analyzed your symptoms and prepared a diagnosis. Please check the prescription details below."
            else:
                response = "I couldn't extract specific symptoms from your description. Could you please provide more details about what you're experiencing?"
    else:
        # General conversation response
        response = "I understand. Can you tell me more about your symptoms? For example, when did they start, how severe are they, and are there any other symptoms you're experiencing?"
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Display prescription section if available
if st.session_state.prescription_data:
    st.divider()
    
    with st.expander("📋 **View Prescription & Diagnosis Details**", expanded=True):
        result = st.session_state.prescription_data["result"]
        symptoms = result["extracted_symptoms"]
        predictions = result["predictions"]

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

        col1, col2 = st.columns([3, 1])
        with col1:
            patient_name = st.text_input(
                "Patient name (for the PDF)",
                value=st.session_state.get("patient_name", ""),
                placeholder="Enter patient name …",
                key="patient_name_input",
            )
            if patient_name:
                st.session_state["patient_name"] = patient_name

        with col2:
            if st.session_state.get("patient_name"):
                pdf_path = detector.generate_pdf(
                    result,
                    patient_name=st.session_state["patient_name"],
                )
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.download_button(
                    label="⬇️  Download PDF",
                    data=pdf_bytes,
                    file_name=f"first_doctor_{st.session_state['patient_name'].replace(' ', '_')}_prescription.pdf",
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
