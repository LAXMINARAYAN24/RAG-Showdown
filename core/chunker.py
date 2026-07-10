from pypdf import PdfReader

def load_and_chunk(pdf_path, chunk_size=500):
    """Read a PDF, return a list of text chunks of ~chunk_size words each."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append(chunk_text)
    return chunks