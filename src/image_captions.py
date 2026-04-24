import os
import json

image_folder = "data/images"

captions = []

for image_name in os.listdir(image_folder):
    caption = {
        "image": image_name,
        "caption": f"This is an image extracted from the IFC report: {image_name}"
    }

    captions.append(caption)

for item in captions:
    print(item)

with open("data/processed/image_captions.json", "w", encoding="utf-8") as f:
    json.dump(captions, f, indent=2)

print("\nSaved image_captions.json")