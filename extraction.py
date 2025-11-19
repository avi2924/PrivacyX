import os
import io
import tempfile
from tqdm import tqdm

# Primary libs
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
import pikepdf
from pdf2image import convert_from_path
import pytesseract

PDF_DIR = "./finance"
TEXT_DIR = "./extracted_texts"
ERR_DIR = "./extract_errors"
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(ERR_DIR, exist_ok=True)

def extract_with_pymupdf(path):
    """Try text extraction using PyMuPDF (fitz). Returns text or ''."""
    try:
        doc = fitz.open(path)
        # If encrypted & requires password, fitz.open will raise
        text_parts = []
        for page in doc:
            # get_text("text") is fine; "blocks" or "dict" could be used for more control
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    except Exception:
        return ""

def repair_pdf_with_pikepdf(path):
    """Try to repair/linearize the PDF using pikepdf and return path to repaired temp file."""
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with pikepdf.open(path) as pdf:
            pdf.save(tmp.name)
        return tmp.name
    except Exception:
        return None

def extract_with_pdfminer(path):
    """Try pdfminer.six text extraction."""
    try:
        text = pdfminer_extract_text(path) or ""
        return text
    except Exception:
        return ""

def ocr_pdf(path, dpi=200):
    """Convert PDF pages to images and OCR them. Returns combined text."""
    try:
        images = convert_from_path(path, dpi=dpi)
        if not images:
            return ""
        ocr_texts = []
        for img in images:
            # pytesseract.image_to_string works with PIL images
            ocr_texts.append(pytesseract.image_to_string(img))
        return "\n".join(ocr_texts)
    except Exception:
        return ""

def is_probably_pdf(path):
    """Quick check for PDF header bytes (not definitive but fast)."""
    try:
        with open(path, "rb") as f:
            header = f.read(5)
        return header.startswith(b"%PDF")
    except Exception:
        return False

def process_pdf_file(pdf_path, out_txt_path, fail_log_path):
    # quick header check (not definitive)
    if not is_probably_pdf(pdf_path):
        with open(fail_log_path, "a", encoding="utf-8") as efile:
            efile.write(f"{pdf_path}: missing %PDF header\n")
        return False

    # 1) Try PyMuPDF
    text = extract_with_pymupdf(pdf_path)
    if text and text.strip():
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True

    # 2) Try repairing with pikepdf then retry pymupdf
    repaired = repair_pdf_with_pikepdf(pdf_path)
    if repaired:
        try:
            text = extract_with_pymupdf(repaired)
            if text and text.strip():
                with open(out_txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                # cleanup temp repaired file
                try:
                    os.remove(repaired)
                except Exception:
                    pass
                return True
        finally:
            # ensure cleanup even on failure
            if os.path.exists(repaired):
                try:
                    os.remove(repaired)
                except Exception:
                    pass

    # 3) Try pdfminer
    text = extract_with_pdfminer(pdf_path)
    if text and text.strip():
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True

    # 4) OCR fallback (slow)
    text = ocr_pdf(pdf_path)
    if text and text.strip():
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True

    # 5) all failed — log
    with open(fail_log_path, "a", encoding="utf-8") as efile:
        efile.write(f"{pdf_path}: all extract methods failed\n")
    return False

def main():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    fail_log = os.path.join(ERR_DIR, "failed_extractions.log")
    for filename in tqdm(pdf_files, desc="Extracting text from PDFs"):
        pdf_path = os.path.join(PDF_DIR, filename)
        text_path = os.path.join(TEXT_DIR, filename.rsplit(".", 1)[0] + ".txt")
        try:
            ok = process_pdf_file(pdf_path, text_path, fail_log)
            if not ok:
                tqdm.write(f"❌ Failed to extract: {filename} (see {fail_log})")
        except Exception as e:
            # unexpected top-level error: log it and continue
            with open(fail_log, "a", encoding="utf-8") as efile:
                efile.write(f"{pdf_path}: unexpected exception: {repr(e)}\n")
            tqdm.write(f"❌ Unexpected error for {filename}: {e}")

if __name__ == "__main__":
    main()
