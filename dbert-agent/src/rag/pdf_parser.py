import logging
from pathlib import Path
from pypdf import PdfReader

logger = logging.getLogger("dbert.rag.pdf_parser")

def extract_text_and_tables(path: str) -> str:
    """
    Extracts text page-by-page from a PDF document.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {path}")
        
    logger.info(f"Extracting text from PDF: {path}")
    text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(f"--- Page {page_idx + 1} ---\n{page_text.strip()}")
    except Exception as e:
        logger.error(f"Error parsing PDF {path}: {e}")
        raise e
        
    return "\n\n".join(text)
