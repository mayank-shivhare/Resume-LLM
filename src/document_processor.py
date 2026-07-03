import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_text(file_path: str) -> str:
    """Load text from a PDF file using PyMuPDF."""
    document = fitz.open(file_path)
    pages = [page.get_text("text") for page in document]
    return "\n\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split long text into smaller chunks for embeddings and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def load_and_chunk_pdf(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Load the PDF text and split it into chunks."""
    text = load_pdf_text(file_path)
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
