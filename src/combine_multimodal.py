import json

# load text chunks
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    text_chunks = json.load(f)

# load table chunks
with open("data/processed/table_chunks.json", "r", encoding="utf-8") as f:
    table_chunks = json.load(f)

# load image captions
with open("data/processed/image_captions.json", "r", encoding="utf-8") as f:
    image_chunks = json.load(f)

combined = []

# add text
for chunk in text_chunks:
    combined.append({
        "type": "text",
        "content": chunk["text"]
    })

# add tables
for chunk in table_chunks:
    combined.append({
        "type": "table",
        "content": chunk["text"]
    })

# add images
for chunk in image_chunks:
    combined.append({
        "type": "image",
        "content": chunk["caption"]
    })

with open("data/processed/multimodal_chunks.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)

print("Saved multimodal_chunks.json")
print("Total chunks:", len(combined))