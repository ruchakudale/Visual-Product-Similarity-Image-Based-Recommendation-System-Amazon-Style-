import matplotlib.pyplot as plt
from PIL import Image
from search_engine import search_similar_products
import pickle, os
from config import EMBEDDING_DIR

# pick a real query image automatically
with open(os.path.join(EMBEDDING_DIR, "image_paths.pkl"), "rb") as f:
    image_paths = pickle.load(f)

query_image = image_paths[10]   # you can change index

results = search_similar_products(query_image, top_k=6)

# ---- Plot ----
plt.figure(figsize=(15,5))

# Show query image
img = Image.open(query_image)
plt.subplot(1,7,1)
plt.imshow(img)
plt.title("QUERY")
plt.axis("off")

# Show results
for i, r in enumerate(results):
    img = Image.open(r["image_path"])
    plt.subplot(1,7,i+2)
    plt.imshow(img)
    plt.title(f"{r['similarity_score']:.2f}")
    plt.axis("off")

plt.show()