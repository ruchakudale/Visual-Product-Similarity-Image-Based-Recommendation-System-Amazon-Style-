import os
import numpy as np
import faiss
import pickle

from config import EMBEDDING_DIR, FAISS_INDEX_DIR

os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

def build_faiss_index():
    print("Loading embeddings...")

    embeddings = np.load(os.path.join(EMBEDDING_DIR, "image_embeddings.npy"))

    # Load image paths (not needed for FAISS, but good check)
    with open(os.path.join(EMBEDDING_DIR, "image_paths.pkl"), "rb") as f:
        image_paths = pickle.load(f)

    print("Embeddings shape:", embeddings.shape)

    # STEP 1: Normalize embeddings (IMPORTANT for cosine similarity)
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # STEP 2: Create FAISS index
    dimension = embeddings.shape[1]  # 2048 for ResNet50
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine)

    print("Building FAISS index...")
    index.add(embeddings)

    print("Total vectors in index:", index.ntotal)

    # STEP 3: Save index
    faiss.write_index(index, os.path.join(FAISS_INDEX_DIR, "product_index.faiss"))

    print("✅ FAISS index saved!")

if __name__ == "__main__":
    build_faiss_index()