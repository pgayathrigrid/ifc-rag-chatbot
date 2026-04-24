import json
table_chunks = []

with open("tables_output.txt", "r", encoding="utf-8") as f:
    content = f.read()

tables = content.split("--------------------------------------------------")

for i, table in enumerate(tables):
    if table.strip():
        table_chunks.append({
            "table_id": i,
            "text": table.strip()
        })

print("Number of table chunks:", len(table_chunks))

for chunk in table_chunks[:3]:
    print("\n", chunk)

with open("data/processed/table_chunks.json", "w", encoding="utf-8") as f:
    json.dump(table_chunks, f, indent=2)

print("\nSaved table_chunks.json")