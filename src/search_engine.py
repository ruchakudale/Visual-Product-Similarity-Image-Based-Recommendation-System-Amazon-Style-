import os
import faiss
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from config import FAISS_INDEX_DIR, EMBEDDING_DIR, IMAGE_SIZE, DEVICE
from model import ResNetFeatureExtractor


# -------------------------------
# Load FAISS index & image paths
# -------------------------------
print("Loading FAISS index...")
index = faiss.read_index(os.path.join(FAISS_INDEX_DIR, "product_index.faiss"))



# Project root: parent of the src folder
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "data" / "images"
PATHS_PATH = BASE_DIR / "data" / "embeddings" / "image_paths.pkl"

with open(PATHS_PATH, "rb") as file:
    saved_paths = pickle.load(file)

# Rebuild paths using the current computer's project directory
image_paths = [
    str(IMAGE_DIR / Path(str(path).replace("\\", "/")).name)
    for path in saved_paths
]

# Check that the images exist
missing = [path for path in image_paths if not Path(path).is_file()]

if missing:
    raise FileNotFoundError(
        f"{len(missing)} product images could not be found. "
        f"Example missing path: {missing[0]}"
    )

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
