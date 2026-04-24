import json

# load metadata chunks
with open("data/processed/chunks_with_metadata.json") as f:
    chunks = json.load(f)

# example: show only Executive Summary chunks
filtered = []

for chunk in chunks:
    if chunk["section"] == "Executive Summary":
        filtered.append(chunk)

print("Found", len(filtered), "chunks in Executive Summary\n")

for chunk in filtered[:3]:
    print("Page:", chunk["page"])
    print(chunk["text"])
    print("\n" + "=" * 60 + "\n")