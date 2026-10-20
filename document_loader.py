"""Modules 1-3: validate PDFs, extract text page by page, clean it, and chunk it."""
import io
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config


def validate_file(filename: str, data: bytes):
    """Return an error message string, or None if the file is acceptable."""
    if not filename.lower().endswith(".pdf"):
        return f"{filename}: only PDF files are allowed."
    if len(data) == 0:
        return f"{filename}: file is empty."
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        return f"{filename}: larger than {config.MAX_FILE_MB} MB."
    if b"%PDF-" not in data[:1024]:
        return f"{filename}: not a valid PDF."
    return None


def clean_text(text: str) -> str:
    """Fix common PDF extraction problems so chunks are cleaner."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)               # re-join hyphenated words
    text = re.sub(r"[ \t]+", " ", text)                    # collapse spaces
    # join single line breaks inside a paragraph, keep bullets / numbered lines
    text = re.sub(r"(?<!\n)\n(?!\n|\s*(?:[•●▪\-\*]|\d+[\.\)])\s)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(filename: str, data: bytes):
    """Extract one Document per page. Returns (documents, total_pages, skipped_pages)."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError(f"{filename}: PDF is password protected.")

    docs, skipped = [], 0
    total = len(reader.pages)
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = clean_text(raw)
        if len(text) < 20:          # empty / scanned / image-only page -> skip safely
            skipped += 1
            continue
        docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
    return docs, total, skipped


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = [c for c in splitter.split_documents(docs) if len(c.page_content.strip()) >= config.MIN_CHUNK_CHARS]
    for n, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = n
    return chunks


def process_files(files):
    """files: list of (filename, bytes). Returns (chunks, report, errors)."""
    all_docs, report, errors = [], [], []
    for name, data in files:
        try:
            docs, total, skipped = load_pdf(name, data)
        except Exception as exc:
            errors.append(f"{name}: could not be read ({exc})")
            continue
        if not docs:
            errors.append(f"{name}: no extractable text (it may be a scanned PDF; OCR is an optional extension).")
            continue
        all_docs.extend(docs)
        report.append({"file": name, "pages": total, "skipped_pages": skipped})

    chunks = split_documents(all_docs)
    counts = Counter(c.metadata["source"] for c in chunks)
    for r in report:
        r["chunks"] = counts.get(r["file"], 0)
    return chunks, report, errors
