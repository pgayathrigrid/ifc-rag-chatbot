import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# load multimodal chunks
with open("data/processed/multimodal_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# load FAISS index
index = faiss.read_index("vectorstores/multimodal.index")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "What was IFC's net income in FY24?"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(query_embedding, 5)

print("\nTop multimodal results:\n")

for i in indices[0]:
    print("Type:", chunks[i]["type"])
    print("Content:")
    print(chunks[i]["content"])
    print("\n" + "=" * 60 + "\n")