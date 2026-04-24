from pypdf import PdfReader 
import json

reader = PdfReader("data/ifc-annual-report-2024-financials.pdf")

all_pages = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()

    all_pages.append({
        "page": i + 1,
        "text": text
    })

with open("data/processed/raw_text.json", "w") as f:
    json.dump(all_pages, f, indent=2)

print("Done! Text extracted.")