import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle

from dataset import ProductImageDataset
from model import load_model
from config import BATCH_SIZE, DEVICE, EMBEDDING_DIR

os.makedirs(EMBEDDING_DIR, exist_ok=True)

def generate_embeddings():
    dataset = ProductImageDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model()

    all_embeddings = []
    all_paths = []

    with torch.no_grad():
        for images, paths in tqdm(loader):
            images = images.to(DEVICE)
            embeddings = model(images)
            embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)
            all_paths.extend(paths)

    all_embeddings = np.vstack(all_embeddings)

    # Save embeddings
    np.save(os.path.join(EMBEDDING_DIR, "image_embeddings.npy"), all_embeddings)

    # Save mapping image ↔ index
    with open(os.path.join(EMBEDDING_DIR, "image_paths.pkl"), "wb") as f:
        pickle.dump(all_paths, f)

    print("✅ Embeddings saved!")

if __name__ == "__main__":
    generate_embeddings()