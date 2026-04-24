import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import os

# load chunks
with open("data/processed/multimodal_chunks.json", "r") as f:
    chunks = json.load(f)

# load FAISS
index = faiss.read_index("vectorstores/multimodal.index")

# embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# sample evaluation questions
questions = [
    "What is IFC?",
    "What is IFC’s mission?",
    "What sectors does IFC support?"
]

for q in questions:
    print("\n==========================")
    print("QUESTION:", q)

    # embedding
    emb = np.array(embed_model.encode([q]), dtype="float32")

    # search
    distances, indices = index.search(emb, 3)

    context = ""
    for i in indices[0]:
        context += chunks[i]["content"] + "\n"

    # prompt
    prompt = f"""
Answer using the context.

Question: {q}

Context:
{context}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("ANSWER:", response.text)