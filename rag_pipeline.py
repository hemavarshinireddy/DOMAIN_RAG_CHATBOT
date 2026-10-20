"""Modules 5-6: retrieve (FAISS + cross-encoder re-ranking) and generate a grounded answer."""
from functools import lru_cache

import config
from prompt import FALLBACK_MESSAGE, build_messages


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


@lru_cache(maxsize=1)
def get_llm():
    provider = config.LLM_PROVIDER
    problem = config.missing_api_key()
    if problem:
        raise RuntimeError(problem)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL, temperature=0, max_tokens=900)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'.")


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return a list of parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class RAGPipeline:
    def __init__(self, vector_store):
        self.vs = vector_store

    # ---------- Module 5: retrieval ----------
    def retrieve(self, question: str):
        """Return [(Document, score)] best-first. Empty list => nothing relevant found."""
        candidates = self.vs.similarity_search(question, k=config.FETCH_K)
        if not candidates:
            return []

        if not config.USE_RERANKER:
            return [(d, 0.0) for d in candidates[: config.TOP_K]]

        scores = get_reranker().predict(
            [(question, d.page_content) for d in candidates], show_progress_bar=False
        )
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(d, float(s)) for d, s in ranked[: config.TOP_K] if s >= config.MIN_RERANK_SCORE]

    @staticmethod
    def _format_context(hits) -> str:
        blocks = []
        for doc, _ in hits:
            m = doc.metadata
            blocks.append(f"[Source: {m['source']} | Page {m['page']}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _unique_sources(hits):
        seen, out = set(), []
        for doc, score in hits:
            key = (doc.metadata["source"], doc.metadata["page"])
            if key in seen:
                continue
            seen.add(key)
            snippet = doc.page_content.strip().replace("\n", " ")
            out.append({
                "source": key[0],
                "page": key[1],
                "snippet": snippet[:300] + ("..." if len(snippet) > 300 else ""),
                "score": round(score, 2),
            })
        return out

    # ---------- Module 6: answer generation ----------
    def answer(self, question: str) -> dict:
        question = question.strip()
        hits = self.retrieve(question)
        retrieved = [(d.metadata["source"], d.metadata["page"]) for d, _ in hits]

        if not hits:
            return {"answer": FALLBACK_MESSAGE, "sources": [], "found": False, "retrieved": retrieved}

        try:
            response = get_llm().invoke(build_messages(question, self._format_context(hits)))
            text = _to_text(response.content).strip()
        except Exception as exc:
            return {
                "answer": f"⚠️ The language model could not be reached: {exc}",
                "sources": self._unique_sources(hits),
                "found": False,
                "retrieved": retrieved,
            }

        found = FALLBACK_MESSAGE.lower().rstrip(".") not in text.lower()
        return {
            "answer": text,
            "sources": self._unique_sources(hits) if found else [],
            "found": found,
            "retrieved": retrieved,
        }
