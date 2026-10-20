"""Module 4: embeddings + FAISS vector store (cosine similarity, saved locally)."""
import json
import shutil
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

import config

_META = config.INDEX_DIR / "meta.json"


@lru_cache(maxsize=1)
def get_embeddings():
    # normalize_embeddings=True -> inner product == cosine similarity
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def index_exists() -> bool:
    return (config.INDEX_DIR / "index.faiss").exists() and (config.INDEX_DIR / "index.pkl").exists()


def build_and_save(chunks, report):
    store = FAISS.from_documents(
        chunks, get_embeddings(), distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    clear_saved()
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.INDEX_DIR))
    _META.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return store


def load_saved():
    if not index_exists():
        return None
    # Safe here: we only load an index that this app itself wrote to local disk.
    return FAISS.load_local(
        str(config.INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def load_meta():
    try:
        return json.loads(_META.read_text(encoding="utf-8"))
    except Exception:
        return []


def clear_saved():
    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
