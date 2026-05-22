import numpy as np
import faiss
import pickle
import os
import random
import pandas as pd

# =========================================
# PATH SETUP
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "image_embeddings.npy")
PATHS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "image_paths.pkl")
FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss_index", "product_index.faiss")
STYLES_PATH = os.path.join(BASE_DIR, "data", "styles.csv")  # <- put csv inside data folder

K = 5
NUM_QUERIES = 200  # evaluate on random subset

# =========================================
# LOAD DATA
# =========================================
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)

print("Loading image paths...")
with open(PATHS_PATH, "rb") as f:
    image_paths = pickle.load(f)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_PATH)

print("Loading metadata (styles.csv)...")
styles = pd.read_csv(STYLES_PATH, on_bad_lines="skip")

print("Total images:", len(image_paths))

# =========================================
# EXTRACT IMAGE ID FROM PATH
# images/12345.jpg → 12345
# =========================================
def get_id_from_path(path):
    return int(os.path.basename(path).split(".")[0])

# create id → metadata lookup
styles["id"] = styles["id"].astype(int)
meta_lookup = styles.set_index("id")

# =========================================
# RELEVANCE FUNCTIONS
# =========================================

def is_relevant_article_type(q_meta, r_meta):
    return q_meta["articleType"] == r_meta["articleType"]

def is_relevant_subcategory(q_meta, r_meta):
    return q_meta["subCategory"] == r_meta["subCategory"]

def is_relevant_strict(q_meta, r_meta):
    return (
        q_meta["articleType"] == r_meta["articleType"]
        and q_meta["baseColour"] == r_meta["baseColour"]
    )

# =========================================
# EVALUATION
# =========================================

def evaluate(metric_function, name):
    precisions = []
    query_indices = random.sample(range(len(embeddings)), NUM_QUERIES)

    for q_idx in query_indices:
        query_embedding = embeddings[q_idx].reshape(1, -1)
        D, I = index.search(query_embedding, K + 1)

        query_id = get_id_from_path(image_paths[q_idx])
        if query_id not in meta_lookup.index:
            continue

        query_meta = meta_lookup.loc[query_id]

        retrieved = I[0][1:]  # remove self match
        relevant_count = 0

        for r_idx in retrieved:
            retrieved_id = get_id_from_path(image_paths[r_idx])
            if retrieved_id not in meta_lookup.index:
                continue

            retrieved_meta = meta_lookup.loc[retrieved_id]

            if metric_function(query_meta, retrieved_meta):
                relevant_count += 1

        precisions.append(relevant_count / K)

    print(f"{name}: {np.mean(precisions):.3f}")

# =========================================
# RUN ALL METRICS
# =========================================

print("\n🎯 REAL VISUAL RETRIEVAL METRICS\n")

evaluate(is_relevant_article_type, "Precision@5 (ArticleType)")
evaluate(is_relevant_subcategory, "Precision@5 (SubCategory)")
evaluate(is_relevant_strict, "Precision@5 (ArticleType + Color)")