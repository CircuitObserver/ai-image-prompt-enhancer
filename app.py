import streamlit as st
from huggingface_hub import InferenceClient
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Prompt Architect", page_icon="🎨", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

# --- API SETUP ---
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

if not hf_token:
    st.error("Missing HF_TOKEN in Secrets.")
    st.stop()

client = InferenceClient(api_key=hf_token)

# --- REFINED SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are an expert AI Image Prompt Engineer. Expand the user input into a professional-grade prompt. "
    "Structure: Subject, Action, Environment, Lighting, Style, Composition, Textures.\n"
    "DO NOT use square brackets []. DO NOT use labels like 'Subject:'. "
    "Provide one continuous, highly descriptive paragraph. No filler."
)

def clean_prompt(text):
    return re.sub(r'[\[\]]', '', text).strip()

# --- UI LAYOUT ---
st.title("🎨 AI Prompt Architect")
st.markdown("Generate and manage high-fidelity image prompts.")

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7)
    if st.button("Clear History"):
        st.session_state.prompt_history = []
        st.rerun()

# --- MAIN INPUT AREA ---
with st.container():
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        user_input = st.text_input("Enter your basic idea:", placeholder="e.g., A cosmic lighthouse", label_visibility="collapsed")
    with col_btn:
        generate_clicked = st.button("Enhance ✨", use_container_width=True, type="primary")

if generate_clicked and user_input:
    with st.spinner("Architecting..."):
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Enhance: {user_input}"}
                ],
                max_tokens=400,
                temperature=temperature
            )
            final_result = clean_prompt(response.choices[0].message.content)
            # Add to history
            st.session_state.prompt_history.insert(0, {"original": user_input, "enhanced": final_result})
        except Exception as e:
            # Check if it's a 401 (token issue) or 429 (rate limit)
            st.error(f"Something went wrong: {e}")

# --- PROMPT HISTORY GRID ---
st.divider()
st.subheader("Prompt History")

if not st.session_state.prompt_history:
    st.info("Your prompts will appear here.")
else:
    for idx, item in enumerate(st.session_state.prompt_history):
        with st.expander(f"Prompt {len(st.session_state.prompt_history) - idx}: {item['original']}", expanded=(idx == 0)):
            # Show the prompt in a code block which has its own built-in copy button!
            # This is the most reliable way in Streamlit currently.
            st.code(item['enhanced'], language="text")
            
            # If you still want a custom button, we use the text_area approach 
            # as it's the most compatible across all browsers.
            st.caption("Click the icon in the top right of the box above to copy.")

st.divider()
st.caption("Running on Python 3.14 | Qwen 2.5-72B")