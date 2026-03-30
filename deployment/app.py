import streamlit as st
from generate_signatures import generate_signatures
import io
import zipfile
import os

# ---------------- App Config ----------------
st.set_page_config(page_title="AI Signature Generator", layout="centered")

st.title("✍ AI Signature Generator (GAN)")
st.write("Generate realistic handwritten signatures using trained GAN model.")

# ---------------- User Input ----------------
num_signatures = st.slider("Select number of signatures", 1, 20, 8)

# ---------------- Generate Signatures ----------------
if st.button("Generate Signatures"):

    with st.spinner("Generating signatures..."):

        # Generate signatures using your GAN model
        image_paths = generate_signatures(
            N=num_signatures,
            output_dir="generated",
            model_path="../checkpoints/G_final.pth"
        )

    st.success(f"Generated {len(image_paths)} signatures!")

    # ---------------- Display Images in Rows ----------------
    cols_per_row = 4
    for i in range(0, len(image_paths), cols_per_row):
        cols = st.columns(cols_per_row)
        row_images = image_paths[i:i + cols_per_row]
        for col, img_path in zip(cols, row_images):
            col.image(img_path, use_container_width=True)

    # ---------------- Create In-Memory ZIP ----------------
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for img_path in image_paths:
            # Add file to zip with just the filename
            zipf.write(img_path, os.path.basename(img_path))

    zip_buffer.seek(0)

    # ---------------- Download Button ----------------
    st.download_button(
        label="⬇ Download All Signatures (ZIP)",
        data=zip_buffer,
        file_name="generated_signatures.zip",
        mime="application/zip"
    )