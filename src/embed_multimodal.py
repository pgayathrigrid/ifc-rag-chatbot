import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# load multimodal chunks
with open("data/processed/multimodal_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# get text content from every chunk
texts = [chunk["content"] for chunk in chunks]

print("Creating embeddings...")

embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

# create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# save index
faiss.write_index(index, "vectorstores/multimodal.index")

print("Saved multimodal.index")
print("Total indexed chunks:", len(chunks))