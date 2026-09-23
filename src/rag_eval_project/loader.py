from pypdf import PdfReader


def load_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file, page by page, concatenated."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text