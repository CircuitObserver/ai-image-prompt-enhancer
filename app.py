import streamlit as st
from huggingface_hub import InferenceClient
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Prompt Architect", page_icon="🎨", layout="centered")

# --- API SETUP ---
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

if not hf_token:
    st.error("Please add your Hugging Face API Token (HF_TOKEN) to Streamlit Secrets.")
    st.stop()

client = InferenceClient(api_key=hf_token)

# --- REFINED SYSTEM PROMPT ---
# We now explicitly tell the model NOT to use brackets in the output.
SYSTEM_PROMPT = (
    "You are an expert AI Image Prompt Engineer. Your task is to take a basic user input "
    "and expand it into a high-detail, professional-grade prompt.\n\n"
    "Structure the output exactly like this, but DO NOT include the square brackets or labels in the final text:\n"
    "Subject + Key Descriptors, Action/Pose/Expression, Environment/Setting + Atmosphere, "
    "Lighting + Shading + Color Scheme, Style/Medium, Composition + Camera Details, Textures + Technical Specs.\n\n"
    "Ensure the result is a single, fluid paragraph of descriptive tags and phrases. "
    "Output ONLY the enhanced prompt. No conversational filler, no headers, and NO square brackets."
)

def clean_prompt(text):
    """
    Post-processing to ensure no brackets remain if the LLM ignores instructions.
    """
    # Removes [ and ] characters
    return re.sub(r'[\[\]]', '', text).strip()

# --- UI LAYOUT ---
st.title("🎨 AI Prompt Architect")
st.markdown("Convert simple ideas into detailed prompts for high-end image generation.")

user_input = st.text_input("Enter your basic idea:", placeholder="e.g., A futuristic cyberpunk cat")

if st.button("Enhance Prompt", type="primary"):
    if user_input:
        with st.spinner("Refining your vision..."):
            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Enhance this prompt: {user_input}"}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                raw_content = response.choices[0].message.content
                # Apply the cleaning function to strip any brackets
                final_prompt = clean_prompt(raw_content)
                
                st.subheader("Your Enhanced Prompt:")
                # Use a text area so users can easily select and copy the text
                st.text_area(label="Copy-paste this into your image generator:", 
                             value=final_prompt, 
                             height=200)
                
                st.success("Prompt generated successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a subject first.")

# --- FOOTER ---
st.divider()
st.caption("2026 Edition | Optimized for Qwen 2.5-72B & Hugging Face Inference Endpoints")