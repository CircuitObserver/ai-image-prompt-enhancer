import streamlit as st
from openai import OpenAI
import os
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Prompt Architect", 
    page_icon="🎨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE (History Management) ---
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

# --- 3. API SETUP (OpenRouter) ---
# Ensure you have OPENROUTER_API_KEY in your Streamlit Secrets
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("⚠️ Missing API Key! Please add 'OPENROUTER_API_KEY' to your Streamlit Secrets.")
    st.stop()

# OpenRouter uses the OpenAI-compatible SDK
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

# --- 4. SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are an expert AI Image Prompt Engineer. Your goal is to expand a basic user idea "
    "into a high-fidelity image generation prompt.\n\n"
    "STRICT STRUCTURE TO FOLLOW:\n"
    "Subject + Key Descriptors, Action/Pose/Expression, Environment/Setting + Atmosphere, "
    "Lighting + Shading + Color Scheme, Style/Medium, Composition + Camera Details, Textures + Quality Enhancers.\n\n"
    "RULES:\n"
    "- DO NOT use square brackets [].\n"
    "- DO NOT use labels like 'Subject:' or 'Environment:'.\n"
    "- Provide ONE continuous, highly descriptive paragraph.\n"
    "- Stay precise to the user's intent but fill in all technical details."
)

def clean_text(text):
    """Safety net to strip any brackets the model might still include."""
    return re.sub(r'[\[\]]', '', text).strip()

# --- 5. UI LAYOUT ---
st.title("🎨 AI Prompt Architect")
st.markdown("Enhance basic ideas into professional image prompts using free 2026 AI models.")

with st.sidebar:
    st.header("Settings")
    # 'openrouter/free' is the most stable way to ensure 100% free usage
    selected_model = st.selectbox(
        "Model Engine",
        ["openrouter/free", "meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen-2.5-72b-instruct:free"],
        help="The 'free' router automatically picks the best available no-cost model."
    )
    
    st.divider()
    if st.button("🗑️ Clear History"):
        st.session_state.prompt_history = []
        st.rerun()

# --- 6. INPUT SECTION ---
with st.container():
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        user_input = st.text_input(
            "Enter your basic idea:", 
            placeholder="e.g., A futuristic cyberpunk cat in Tokyo", 
            label_visibility="collapsed"
        )
    with col_btn:
        btn_label = "Enhance ✨"
        generate_clicked = st.button(btn_label, use_container_width=True, type="primary")

# --- 7. LOGIC & API CALL ---
if generate_clicked and user_input:
    with st.spinner("Architecting your prompt..."):
        try:
            completion = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Enhance this prompt: {user_input}"}
                ],
                # Recommended headers for OpenRouter
                extra_headers={
                    "HTTP-Referer": "https://streamlit.io", 
                    "X-Title": "AI Prompt Architect"
                }
            )
            
            raw_response = completion.choices[0].message.content
            final_prompt = clean_text(raw_response)
            
            # Store in history (newest first)
            st.session_state.prompt_history.insert(0, {
                "original": user_input, 
                "enhanced": final_prompt
            })
            
        except Exception as e:
            st.error(f"Error: {e}")

# --- 8. PROMPT HISTORY GRID ---
st.divider()
st.subheader("Prompt History")

if not st.session_state.prompt_history:
    st.info("Your enhanced prompts will appear here as you create them.")
else:
    for idx, item in enumerate(st.session_state.prompt_history):
        # We use st.code for the 'Enhanced' version because it has a native COPY button
        with st.expander(f"📌 {item['original']}", expanded=(idx == 0)):
            st.markdown("**Enhanced Prompt:**")
            st.code(item['enhanced'], language="text")
            st.caption("Hover over the box above and click the icon in the top right to copy.")

st.divider()
st.caption("Powered by OpenRouter Free Tier | May 2026 Edition")