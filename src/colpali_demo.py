import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load page-level data
# -----------------------------
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

# group by page (simulate full-page retrieval)
pages = {}

for chunk in chunks:
    page = chunk["page"]
    if page not in pages:
        pages[page] = ""
    pages[page] += chunk["text"] + " "

page_texts = list(pages.values())
page_ids = list(pages.keys())

# -----------------------------
# Embeddings
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

page_embeddings = np.array(
    model.encode(page_texts),
    dtype="float32"
)

# -----------------------------
# Build FAISS index
# -----------------------------
index = faiss.IndexFlatL2(page_embeddings.shape[1])
index.add(page_embeddings)

# -----------------------------
# Query
# -----------------------------
query = "What is IFC mission?"

query_embedding = np.array(
    model.encode([query]),
    dtype="float32"
)

# search top pages
distances, indices = index.search(query_embedding, 3)

print("\nTop relevant pages:\n")

for idx in indices[0]:
    print("PAGE:", page_ids[idx])
    print(page_texts[idx][:500])
    print("\n" + "=" * 60 + "\n")