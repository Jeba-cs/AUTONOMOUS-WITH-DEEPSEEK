# pdf_utils.py

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split text into overlapping chunks using LangChain's splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.create_documents([text])
