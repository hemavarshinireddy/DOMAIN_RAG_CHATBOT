"""Central configuration. Every value can be overridden from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "vector_store" / "saved_index"


def _int(name, default):
    return int(os.getenv(name, default))


def _float(name, default):
    return float(os.getenv(name, default))


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


# ---------- Upload limits (Module 1 / Security) ----------
MAX_FILE_MB = _int("MAX_FILE_MB", 20)
MAX_FILES = _int("MAX_FILES", 5)

# ---------- Chunking (Module 3) ----------
CHUNK_SIZE = _int("CHUNK_SIZE", 900)        # brief suggests 700-1000
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)  # brief suggests 100-150
MIN_CHUNK_CHARS = 40

# ---------- Embeddings (Module 4) ----------
# Beginner model from the brief. For better accuracy try: BAAI/bge-small-en-v1.5
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------- Retrieval (Module 5) ----------
FETCH_K = _int("FETCH_K", 20)   # candidates pulled from FAISS
TOP_K = _int("TOP_K", 5)        # final chunks sent to the LLM (brief: 3-5)
USE_RERANKER = _bool("USE_RERANKER", True)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Chunks scoring below this are treated as "not relevant" -> chatbot refuses.
# Lower it (e.g. -5) if it refuses too often; raise it (e.g. -1) if it answers too often.
MIN_RERANK_SCORE = _float("MIN_RERANK_SCORE", -3.0)

# ---------- LLM (Module 6) ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # groq | gemini | openai
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_KEY_NAMES = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}


def missing_api_key():
    """Return a human-readable problem string, or None if the key is set."""
    if LLM_PROVIDER not in _KEY_NAMES:
        return f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use groq, gemini or openai."
    key = _KEY_NAMES[LLM_PROVIDER]
    if not os.getenv(key):
        return f"{key} is not set. Add it to your .env file."
    return None
