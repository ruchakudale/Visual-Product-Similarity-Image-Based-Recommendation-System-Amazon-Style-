import streamlit as st
from PIL import Image
import tempfile
import sys
import os

# allow streamlit to access src modules
sys.path.append("src")

from search_engine import search_similar_products

st.set_page_config(
    page_title="Visual Product Search",
    layout="wide"
)
st.title("🛍️ Visual Product Similarity Search")

st.sidebar.header("Search Settings")

top_k = st.sidebar.slider(
    "Number of similar products",
    min_value=3,
    max_value=12,
    value=6
)

st.write("Upload a product image and find visually similar items.")

uploaded_file = st.file_uploader("Upload product image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # ⭐ FIX: read file correctly using PIL first
    image = Image.open(uploaded_file).convert("RGB")

    # Save to temp file so FAISS pipeline can use path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file.name)

    query_image_path = temp_file.name

    st.subheader("📷 Uploaded Image")
    st.image(image, width=250)

    if st.button("🔍 Find Similar Products"):
        with st.spinner("Searching similar products..."):
            results = search_similar_products(query_image_path, top_k=top_k)

        st.subheader("🧡 Similar Products")

        # cols = st.columns(top_k)

        # for col, result in zip(cols, results):
        #     col.image(result["image_path"])
        #     col.progress(float(result["similarity_score"]))
        #     col.caption(f"{result['similarity_score']:.2f} similarity")
            
        cols = st.columns(4)

        for i, result in enumerate(results):
            with cols[i % 4]:
                st.image(result["image_path"], use_container_width=True)
                st.progress(float(result["similarity_score"]))
                st.caption(f"{result['similarity_score']:.2f} similarity")

        st.markdown("---")
        st.caption("Visual similarity powered by Deep Learning + FAISS")