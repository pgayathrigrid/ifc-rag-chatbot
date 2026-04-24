import json
import re

with open("data/processed/raw_text.json") as f:
    pages = json.load(f)

chunks = []

for page in pages:
    text = page["text"]

    # clean extra spaces and line breaks
    cleaned = re.sub(r"\s+", " ", text).strip()

    # split into chunks of 1000 characters
    chunk_size = 1000

    for i in range(0, len(cleaned), chunk_size):
        chunk = cleaned[i:i + chunk_size]

        chunks.append({
            "page": page["page"],
            "chunk_id": f"page_{page['page']}_chunk_{i}",
            "text": chunk
        })

with open("data/processed/chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

print("Done! Chunks created.")