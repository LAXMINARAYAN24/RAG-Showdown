# core/ocr_chunker.py
import pytesseract
from pdf2image import convert_from_path

from core.chunker import chunk_words

# Point pytesseract to where you installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path to poppler's bin folder from step 2
POPPLER_PATH = r"C:\Users\sahul\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin"

def ocr_pdf_to_chunks(pdf_path, chunk_size=200, overlap=50):
    """Convert each page of a scanned/image PDF into text using OCR, then chunk it."""
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    full_text = ""
    for i, page_image in enumerate(pages):
        print(f"  OCR-ing page {i+1}/{len(pages)}...")
        page_text = pytesseract.image_to_string(page_image)
        full_text += page_text + "\n"

    return chunk_words(full_text.split(), chunk_size=chunk_size, overlap=overlap)