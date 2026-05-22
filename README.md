# 🛍️ Visual Product Similarity & Image-Based Recommendation System (Amazon-Style)

An end-to-end Computer Vision–based recommendation system that retrieves visually similar fashion products using Deep Learning embeddings and FAISS similarity search.

This project replicates the core idea behind Amazon’s **“Similar Items”** and **image-based product search** systems.

---

# 🚀 Project Demo

### Features
✅ Upload a fashion product image  
✅ Retrieve visually similar products instantly  
✅ Deep Learning–based image embeddings  
✅ FAISS-powered similarity search  
✅ Streamlit interactive web application  
✅ Precision@K evaluation metrics  
✅ Scalable architecture for large product catalogs

---

# 📌 Problem Statement

Traditional keyword-based product search often fails because:
- Product metadata can be noisy or incomplete
- Users may search visually rather than textually
- Similar products may not share keywords

This project solves the problem using:
- Computer Vision
- Deep Learning embeddings
- Vector similarity search

---

# 🎯 Business Use Case

Companies like Amazon, Myntra, Flipkart, IKEA, and Pinterest use similar systems to:

- Improve product discovery
- Enable image-based search
- Increase conversion rate
- Recommend visually similar products
- Reduce dependency on manual tagging

---

# 🧠 Solution Overview

The system works in 5 major stages:

1. Image preprocessing
2. Feature extraction using ResNet50
3. Embedding generation
4. FAISS vector indexing
5. Similarity retrieval + Streamlit UI

---

# 🏗️ Project Architecture

## Architecture Flow

```text
Dataset Images
      ↓
Preprocessing (Resize + Normalize)
      ↓
ResNet50 Feature Extraction
      ↓
2048-d Embedding Vectors
      ↓
FAISS Vector Index
      ↓
Similarity Search
      ↓
Top-K Similar Product Retrieval
      ↓
Streamlit Web Application
```

---

# 📂 Project Structure

```text
VISUAL-PRODUCT-RECOMMENDER/
│
├── data/
│   ├── embeddings/
│   │   ├── image_embeddings.npy
│   │   ├── image_paths.pkl
│   │
│   ├── faiss_index/
│   │   ├── product_index.faiss
│   │
│   ├── images/
│   │
│   ├── styles.csv
│
├── src/
│   ├── build_faiss_index.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── extract_embeddings.py
│   ├── model.py
│   ├── search_engine.py
│   ├── test_search.py
│   ├── visualize_results.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
│
└── venv/
```

---

# 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core Programming |
| PyTorch | Deep Learning |
| ResNet50 | Feature Extraction |
| FAISS | Similarity Search |
| NumPy | Numerical Computation |
| Pandas | Metadata Processing |
| PIL/OpenCV | Image Processing |
| Streamlit | Web Application |

---

# 📊 Dataset

## Dataset Used
Fashion Product Dataset containing:

- 44,000+ product images
- Product metadata (`styles.csv`)

### Metadata Fields
- gender
- masterCategory
- subCategory
- articleType
- baseColour
- season
- usage
- productDisplayName

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/ruchakudale/Visual-Product-Similarity-Image-Based-Recommendation-System-Amazon-Style-.git

cd visual-product-recommender
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Required Libraries

```txt
torch
torchvision
faiss-cpu
numpy
pandas
pillow
matplotlib
streamlit
scikit-learn
```

---

# 🔥 Workflow

# Step 1 — Extract Image Embeddings

Run:

```bash
python src/extract_embeddings.py
```

This:
- Loads images
- Uses pretrained ResNet50
- Extracts 2048-d embeddings
- Saves embeddings

Generated files:

```text
data/embeddings/image_embeddings.npy
data/embeddings/image_paths.pkl
```

---

# Step 2 — Build FAISS Index

Run:

```bash
python src/build_faiss_index.py
```

This:
- Loads embeddings
- Creates FAISS ANN index
- Stores vector database

Generated file:

```text
data/faiss_index/product_index.faiss
```

---

# Step 3 — Test Similarity Search

Run:

```bash
python src/test_search.py
```

Example output:

```text
Top similar products:

10010.jpg | score: 1.00
10022.jpg | score: 0.93
7483.jpg  | score: 0.92
```

---

# Step 4 — Visualize Results

Run:

```bash
python src/visualize_results.py
```

Displays:
- Query image
- Top-K visually similar products

---

# Step 5 — Launch Streamlit App

Run:

```bash
streamlit run streamlit_app.py
```

Features:
- Upload image
- Retrieve similar products
- Similarity scores
- Product grid layout

---

# 🧠 Deep Learning Model

## ResNet50

Used pretrained ImageNet weights.

### Why ResNet50?
- Strong visual feature extraction
- Robust transfer learning
- Industry-standard architecture

### Embedding Dimension
```text
2048
```

---

# ⚡ FAISS Similarity Search

FAISS is used for:
- Approximate nearest neighbor search
- Fast vector retrieval
- Scalable similarity matching

### Distance Metric
```text
Cosine Similarity
```

---

# 📈 Evaluation Metrics

The system was evaluated using metadata from `styles.csv`.

## Precision@5 Results

| Metric | Score |
|---|---|
| Precision@5 (SubCategory) | 0.916 |
| Precision@5 (ArticleType) | 0.794 |
| Precision@5 (ArticleType + Color) | 0.301 |

---

# 📌 Interpretation

### Precision@5 (SubCategory) = 0.916
91.6% of retrieved products belong to the same subcategory.

### Precision@5 (ArticleType) = 0.794
~80% of retrieved products belong to same product type.

### Precision@5 (ArticleType + Color) = 0.301
Strict metric combining both product type and exact color.

---

# 🎨 Streamlit Application

## Features
✅ Upload fashion product image  
✅ Top-K similar product retrieval  
✅ Similarity score visualization  
✅ Interactive UI  
✅ Fast image search  

---

# 🏆 Key Achievements

✅ Processed 44k+ product images  
✅ Built scalable ANN retrieval pipeline  
✅ Achieved high Precision@K metrics  
✅ Built production-style architecture  
✅ Created Amazon-style visual recommendation system  

---

# 🚀 Future Improvements

- EfficientNet / CLIP embeddings
- Hybrid image + text search
- Product filtering
- GPU optimization
- Cloud deployment
- Real-time webcam search
- Recommendation personalization

---

# 🧠 Learning Outcomes

- Deep Learning for Computer Vision
- Embedding-based Retrieval Systems
- Vector Databases (FAISS)
- Similarity Search
- Streamlit App Development
- Image Recommendation Systems

---

# 📜 License

This project is licensed under the MIT License.

---

# 🙌 Acknowledgements

- PyTorch
- FAISS by Facebook AI
- Streamlit
- Fashion Product Dataset Contributors

---

# ⭐ If you found this useful

Please consider giving this repository a ⭐ on GitHub.
