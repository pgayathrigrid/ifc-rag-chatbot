import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi  # NEW

# -----------------------------
# Load chunks
# -----------------------------
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# -----------------------------
# BM25 setup (NEW)
# -----------------------------
tokenized_corpus = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized_corpus)

# -----------------------------
# FAISS setup
# -----------------------------
index = faiss.read_index("vectorstores/faiss_index.index")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Query
# -----------------------------
query = "What is IFC's mission to end poverty and increase prosperity?"

# -----------------------------
# FAISS search (semantic)
# -----------------------------
query_embedding = np.array(model.encode([query]), dtype="float32")
distances, indices = index.search(query_embedding, 5)

faiss_indices = list(indices[0])

# -----------------------------
# BM25 search (keyword)
# -----------------------------
tokenized_query = query.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)

bm25_indices = np.argsort(bm25_scores)[-5:]

# -----------------------------
# Hybrid combination
# -----------------------------
combined_indices = list(set(faiss_indices + list(bm25_indices)))

print("\nTop HYBRID matching chunks:\n")

for i in combined_indices[:5]:
    print("PAGE:", chunks[i]["page"])
    print("CHUNK ID:", chunks[i]["chunk_id"])
    print()
    print(chunks[i]["text"])
    print("\n" + "=" * 60 + "\n")