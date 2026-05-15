import streamlit as st
from huggingface_hub import InferenceClient
from st_copy_button import st_copy_button  # Import the new component
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
    "Provide one continuous, highly descriptive paragraph."
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
        user_input = st.text_input("Enter your basic idea:", placeholder="e.g., A neon samurai", label_visibility="collapsed")
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
            st.session_state.prompt_history.insert(0, {"original": user_input, "enhanced": final_result})
        except Exception as e:
            st.error(f"API Error: {e}")

# --- PROMPT HISTORY GRID ---
st.divider()
st.subheader("Prompt History")

if not st.session_state.prompt_history:
    st.info("Your prompts will appear here.")
else:
    for idx, item in enumerate(st.session_state.prompt_history):
        # Using a unique key for each button is vital in Streamlit loops
        with st.expander(f"Prompt {len(st.session_state.prompt_history) - idx}: {item['original']}", expanded=(idx == 0)):
            grid_col_text, grid_col_btn = st.columns([9, 1])
            
            with grid_col_text:
                st.write(item['enhanced'])
            
            with grid_col_btn:
                # This replaces the broken st.copy_to_clipboard
                st_copy_button(
                    text=item['enhanced'],
                    label="📋",
                    key=f"copy_{idx}" # Unique key for each instance
                )

st.divider()
st.caption("2026 Edition | Optimized for Qwen 2.5-72B")