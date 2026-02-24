import streamlit as st
from groq import Groq  # or 'openai' library
import json
from triage import check_emergency

# 1. Configuration
client = Groq(api_key="")
# with open('test_bundles.json', 'r') as f:
#     TEST_DATA = json.load(f)

st.set_page_config(page_title="First Doctor AI", page_icon="🩺")

# 2. Session State (The "Memory" of the website)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are First Doctor. Ask 3 follow-up questions to help triage the user. Be concise."}]
if "step" not in st.session_state:
    st.session_state.step = 0

st.title("🩺 First Doctor")
st.caption("Student Project: AI Triage & Preliminary Diagnostics")

# 3. Chat Interface
for message in st.session_state.messages[1:]: # Hide the system prompt
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your disease..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Safety Check
    # if check_emergency(prompt):
    #     st.error("🚨 EMERGENCY: Please seek immediate medical attention!")
    #     st.stop()

    # Call Cloud Llama API
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", # Use the cloud model name
            messages=st.session_state.messages,
        )
        answer = response.choices[0].message.content
        st.markdown(answer)
        print(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.step += 1

# 4. Final Prescription Logic (Appears after 3 turns)
if st.session_state.step >= 3:
    st.divider()
    st.subheader("📋 Preliminary Test Prescription")
    
    # Combine conversation for keyword matching
    full_text = " ".join([m["content"] for m in st.session_state.messages])
    
    suggested = []
    # for cat, details in TEST_DATA.items():
    #     if any(key in full_text.lower() for key in details['keywords']):
    #         suggested.extend(details['tests'])
    
    if suggested:
        for test in list(set(suggested)):
            st.write(f"- {test}")
        
        # Create a "Download" button for the test list
        prescription_text = "FIRST DOCTOR TEST LIST\n\n" + "\n".join(suggested)
        st.download_button("Download Test Prescription", prescription_text, file_name="tests.txt")
    else:
        st.info("Please consult a GP for a general checkup.")