import json

# load chunks
with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

for chunk in chunks:
    text = chunk["text"]

    # default metadata
    chunk["content_type"] = "text"
    chunk["section"] = "unknown"

    # detect section names
    if "EXECUTIVE SUMMARY" in text:
        chunk["section"] = "Executive Summary"
    elif "OVERVIEW" in text:
        chunk["section"] = "Overview"
    elif "CLIENT SERVICES" in text:
        chunk["section"] = "Client Services"
    elif "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS" in text:
        chunk["section"] = "Financial Notes"

# save new file
with open("data/processed/chunks_with_metadata.json", "w") as f:
    json.dump(chunks, f, indent=2)

print("Saved chunks_with_metadata.json")