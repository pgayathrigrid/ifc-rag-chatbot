import json
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np

# load chunks
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
embeddings = model.encode(texts)

# convert to numpy array
embeddings = np.array(embeddings).astype("float32")

# create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# add embeddings to index
index.add(embeddings)

# save index
faiss.write_index(index, "vectorstores/faiss_index.index")

print("Done! FAISS index created.")