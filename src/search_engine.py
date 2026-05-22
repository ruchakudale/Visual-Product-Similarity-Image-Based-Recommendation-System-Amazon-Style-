import os
import faiss
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms

from config import FAISS_INDEX_DIR, EMBEDDING_DIR, IMAGE_SIZE, DEVICE
from model import ResNetFeatureExtractor


# -------------------------------
# Load FAISS index & image paths
# -------------------------------
print("Loading FAISS index...")
index = faiss.read_index(os.path.join(FAISS_INDEX_DIR, "product_index.faiss"))

print("Loading image paths...")
with open(os.path.join(EMBEDDING_DIR, "image_paths.pkl"), "rb") as f:
    image_paths = pickle.load(f)

print("Loading feature extractor...")
model = ResNetFeatureExtractor()
model = model.to(DEVICE)  
model.eval()


# Image preprocessing (same as training!)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------------
# Convert query image → embedding vector
# -----------------------------------------
def get_query_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model(image)

    embedding = embedding.cpu().numpy()
    faiss.normalize_L2(embedding)

    return embedding


# -----------------------------------------
# Search similar products
# -----------------------------------------
def search_similar_products(query_image_path, top_k=5):
    query_embedding = get_query_embedding(query_image_path)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        results.append({
            "image_path": image_paths[idx],
            "similarity_score": float(score)
        })

    return results