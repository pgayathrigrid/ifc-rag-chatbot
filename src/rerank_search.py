import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# load chunks
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

# load FAISS index
index = faiss.read_index("vectorstores/faiss_index.index")

# model for embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# model for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# question
query = "What is IFC's mission?"

# first search using FAISS
query_embedding = embed_model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(query_embedding, 10)

# get candidate chunks
candidate_chunks = [chunks[i] for i in indices[0]]

# create pairs: [question, chunk]
pairs = []

for chunk in candidate_chunks:
    pairs.append([query, chunk["text"]])

# rerank scores
scores = reranker.predict(pairs)

# combine scores and chunks
ranked_results = list(zip(scores, candidate_chunks))

# sort highest score first
ranked_results.sort(reverse=True, key=lambda x: x[0])

print("\nTop reranked chunks:\n")

for score, chunk in ranked_results:
    print("Score:", score)
    print("Page:", chunk["page"])
    print()
    print(chunk["text"])
    print("\n" + "=" * 60 + "\n")