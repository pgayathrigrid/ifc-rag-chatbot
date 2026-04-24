from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

query = "What was IFC's net income in FY24?"
embedding = model.encode([query])

np.save("data/processed/sample_query_embedding.npy", embedding)
print("Saved demo embedding")