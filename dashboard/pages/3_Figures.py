import streamlit as st
import os

st.set_page_config(page_title="Generated Figures", page_icon="🖼️", layout="wide")

st.title("🖼️ Generated Figures")
st.markdown("View all the plots generated during the evaluation phase.")

FIGURES_DIR = "outputs/figures/"

if os.path.exists(FIGURES_DIR):
    # Automatically load every PNG
    png_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')]
    
    if png_files:
        st.info(f"Found {len(png_files)} figures.")
        
        # Display inside expandable cards
        for file in sorted(png_files):
            # Create a clean title from the filename (e.g., 'reward_comparison.png' -> 'Reward Comparison')
            title = file.replace('.png', '').replace('_', ' ').title()
            
            with st.expander(title):
                img_path = os.path.join(FIGURES_DIR, file)
                st.image(img_path, caption=title, width="stretch")
    else:
        st.warning("No PNG figures found in the directory.")
else:
    st.error(f"Figures directory not found at `{FIGURES_DIR}`. Please run the evaluation first.")
