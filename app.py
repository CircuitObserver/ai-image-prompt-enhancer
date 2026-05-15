import streamlit as st
from openai import OpenAI
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Prompt Architect", page_icon="🎨", layout="wide")

if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

# --- API SETUP (OpenRouter) ---
# Replace HF_TOKEN with OPENROUTER_API_KEY in your Streamlit Secrets
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("Please add 'OPENROUTER_API_KEY' to your Secrets.")
    st.stop()

# OpenRouter uses the OpenAI-compatible SDK
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

SYSTEM_PROMPT = (
    "You are an expert AI Image Prompt Engineer. Expand the user input into a professional-grade prompt. "
    "Structure: Subject, Action, Environment, Lighting, Style, Composition, Textures. "
    "DO NOT use square brackets [] or labels. Provide one continuous, descriptive paragraph."
)

# --- UI ---
st.title("🎨 AI Prompt Architect (Free Edition)")

with st.sidebar:
    st.header("Settings")
    # Using a free-tier model from OpenRouter
    model_choice = st.selectbox("Select Model", 
                                ["qwen/qwen-2.5-72b-instruct:free", "google/gemini-2.0-flash-exp:free"])
    if st.button("Clear History"):
        st.session_state.prompt_history = []
        st.rerun()

with st.container():
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        user_input = st.text_input("Enter idea:", placeholder="e.g., A steampunk owl", label_visibility="collapsed")
    with col_btn:
        generate_clicked = st.button("Enhance ✨", use_container_width=True, type="primary")

if generate_clicked and user_input:
    with st.spinner("Generating with Free Tier..."):
        try:
            completion = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Enhance: {user_input}"}
                ]
            )
            final_result = completion.choices[0].message.content.strip()
            st.session_state.prompt_history.insert(0, {"original": user_input, "enhanced": final_result})
        except Exception as e:
            st.error(f"Error: {e}")

# --- HISTORY ---
st.divider()
for idx, item in enumerate(st.session_state.prompt_history):
    with st.expander(f"{item['original']}", expanded=(idx == 0)):
        st.code(item['enhanced'], language="text")