import streamlit as st
from huggingface_hub import InferenceClient
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Prompt Architect", page_icon="🎨")

# --- API SETUP ---
# On Streamlit Cloud, add HF_TOKEN to "Settings > Secrets"
# Locally, it will look for it in your environment variables
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

if not hf_token:
    st.error("Please add your Hugging Face API Token to continue.")
    st.stop()

client = InferenceClient(api_key=hf_token)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "You are an expert AI Image Prompt Engineer. Your task is to take a basic user input "
    "and expand it into a high-detail, professional-grade prompt for image generation models. "
    "You MUST follow this exact structure for every response:\n\n"
    "[Subject + Key Descriptors], [Action/Pose/Expression], [Environment/Setting + Atmosphere], "
    "[Lighting + Shading + Color Scheme/Gradients + Mood], [Style/Medium + Artistic References], "
    "[Composition + Perspective + Camera/Lens Details], [Textures/Materials + Quality Enhancers + Technical Specs]\n\n"
    "Stay precise to what the user requested but ensure every bracketed section is filled with "
    "vivid, descriptive detail. Do not include any conversational filler; output ONLY the enhanced prompt."
)

# --- UI LAYOUT ---
st.title("🎨 AI Prompt Architect")
st.markdown("Expand simple ideas into professional-grade image prompts using **Qwen 2.5-72B**.")

user_input = st.text_input("Enter your basic idea:", placeholder="e.g., A cat")

if st.button("Enhance Prompt"):
    if user_input:
        with st.spinner("Qwen is architecting your prompt..."):
            try:
                # Using the latest 2026 Inference Client chat completion method
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Enhance this prompt: {user_input}"}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                enhanced_prompt = response.choices[0].message.content
                
                st.subheader("Enhanced Prompt:")
                st.code(enhanced_prompt, language="text")
                st.button("Copy to Clipboard", on_click=lambda: st.write(f"Prompt copied! (Manual copy required in some browsers)"))
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a prompt first!")

# --- FOOTER ---
st.divider()
st.caption("Powered by Qwen/Qwen2.5-72B-Instruct via Hugging Face Inference.")