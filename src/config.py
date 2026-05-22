import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

EMBEDDING_DIR = os.path.join(DATA_DIR, "embeddings")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

BATCH_SIZE = 32
IMAGE_SIZE = 224
EMBEDDING_DIM = 2048  # ResNet50 output
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"