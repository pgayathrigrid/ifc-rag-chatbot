import fitz
import os

pdf_path = "data/ifc-annual-report-2024-financials.pdf"
output_folder = "data/images"

os.makedirs(output_folder, exist_ok=True)

doc = fitz.open(pdf_path)

image_count = 0

for page_num in range(len(doc)):
    page = doc[page_num]

    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)

        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        image_filename = f"page_{page_num + 1}_image_{img_index + 1}.{image_ext}"
        image_path = os.path.join(output_folder, image_filename)

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        print(f"Saved: {image_filename}")
        image_count += 1

print(f"\nTotal images extracted: {image_count}")