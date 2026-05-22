import pickle
import os
from search_engine import search_similar_products
from config import EMBEDDING_DIR

# Load real image paths saved during embedding step
with open(os.path.join(EMBEDDING_DIR, "image_paths.pkl"), "rb") as f:
    image_paths = pickle.load(f)

# Pick first image as query
query_image = image_paths[10]

print("Query image:", query_image)

results = search_similar_products(query_image, top_k=5)

print("\nTop similar products:\n")
for r in results:
    print(r["image_path"], "| score:", round(r["similarity_score"], 4))