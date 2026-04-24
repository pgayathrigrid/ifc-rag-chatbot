import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
import os 

# load chunks
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

# load FAISS index
index = faiss.read_index("vectorstores/faiss_index.index")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

query = "What is IFC's mission?"

# search
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(query_embedding, 3)

context = ""

for i in indices[0]:
    context += chunks[i]["text"] + "\n\n"

prompt = f"""
Answer the question using only the context below.

Question:
{query}

Context:
{context}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\nANSWER:\n")
print(response.text)