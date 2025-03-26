# scripts/01_extract_pdfs.py
import os
import fitz

input_folder = "data/pdfs"
output_folder = "data/raw"
os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.endswith(".pdf"):
        path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname.replace(".pdf", ".txt"))
        with fitz.open(path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

print("✅ Extracted all PDFs into plain text.")
