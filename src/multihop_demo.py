import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
import os

# load data
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

index = faiss.read_index("vectorstores/faiss_index.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# multi-hop question
query = "What is IFC's mission and how many member countries does it have?"

# step 1: embed query
query_embedding = np.array(model.encode([query]), dtype="float32")

# step 2: retrieve MORE chunks (important for multi-hop)
distances, indices = index.search(query_embedding, 8)

context = ""

print("\nRetrieved chunks:\n")

for i in indices[0]:
    print(chunks[i]["text"][:200])
    print("----")
    context += chunks[i]["text"] + "\n"

# step 3: generate combined answer
prompt = f"""
Answer the question using the context. Combine information if needed.

Question: {query}

Context:
{context}

Answer in 2-3 sentences.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\nFINAL ANSWER:\n")
print(response.text)