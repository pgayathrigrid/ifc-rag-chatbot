import fitz  # PyMuPDF

pdf_path = "data/ifc-annual-report-2024-financials.pdf"

doc = fitz.open(pdf_path)

for page_num in range(len(doc)):
    page = doc[page_num]

    tables = page.find_tables()

    if tables.tables:
        print(f"\nPage {page_num + 1}")

        for i, table in enumerate(tables.tables):
            data = table.extract()
            with open("data/tables/tables_output.txt", "a", encoding="utf-8") as f:
                f.write(f"\nPage {page_num + 1}\n")
                f.write(f"Table {i + 1}:\n")
                f.write(str(data))
                f.write("\n" + "-" * 50 + "\n")