import os
import fitz  # PyMuPDF
from tqdm import tqdm

# Input and output directories
PDF_DIR = "./ico_pdfs"         
TEXT_DIR = "./extracted_texts"
os.makedirs(TEXT_DIR, exist_ok=True)

# Iterate through all PDF files in the folder
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

for filename in tqdm(pdf_files, desc="Extracting text from PDFs"):
    try:
        pdf_path = os.path.join(PDF_DIR, filename)
        text_path = os.path.join(TEXT_DIR, filename.replace(".pdf", ".txt"))

        doc = fitz.open(pdf_path)
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    except Exception as e:
        print(f"❌ Failed to extract {filename}: {e}")
