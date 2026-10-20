"""Module 7: Streamlit interface.  Run with:  streamlit run app.py"""
import streamlit as st

import config
import vector_store
from document_loader import process_files, validate_file
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Domain RAG Chatbot", page_icon="📄", layout="wide")


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("report", [])
    st.session_state.setdefault("loaded_saved", False)
    st.session_state.setdefault("uploader_key", 0)

    # Re-use a previously saved index so users don't have to re-process every time.
    if not st.session_state.loaded_saved:
        st.session_state.loaded_saved = True
        with st.spinner("Loading saved index..."):
            store = vector_store.load_saved()
        if store is not None:
            st.session_state.pipeline = RAGPipeline(store)
            st.session_state.report = vector_store.load_meta()


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**📄 {s['source']}** — page {s['page']}")
            st.caption(s["snippet"])


def sidebar():
    st.sidebar.header("📁 Documents")
    problem = config.missing_api_key()
    if problem:
        st.sidebar.error(problem)

    uploaded = st.sidebar.file_uploader(
        f"Upload PDF files (max {config.MAX_FILES} files, {config.MAX_FILE_MB} MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded:
        st.sidebar.markdown("**Selected files:**")
        for f in uploaded:
            st.sidebar.write(f"• {f.name}")

    if st.sidebar.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if not uploaded:
            st.sidebar.warning("Please choose at least one PDF first.")
        elif len(uploaded) > config.MAX_FILES:
            st.sidebar.error(f"Please upload at most {config.MAX_FILES} files.")
        else:
            valid, errors = [], []
            for f in uploaded:
                data = f.getvalue()
                err = validate_file(f.name, data)
                (errors if err else valid).append(err or (f.name, data))
            with st.spinner("Reading, chunking and indexing documents..."):
                chunks, report, load_errors = process_files(valid) if valid else ([], [], [])
                errors += load_errors
                if chunks:
                    store = vector_store.build_and_save(chunks, report)
                    st.session_state.pipeline = RAGPipeline(store)
                    st.session_state.report = report
                    st.session_state.messages = []  # new documents -> fresh chat
            for e in errors:
                st.sidebar.error(e)
            if chunks:
                st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(report)} document(s).")

    if st.session_state.report:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Indexed documents**")
        for r in st.session_state.report:
            skipped = f", {r['skipped_pages']} empty page(s) skipped" if r["skipped_pages"] else ""
            st.sidebar.caption(f"📄 {r['file']} — {r['pages']} pages, {r['chunks']} chunks{skipped}")

    if st.sidebar.button("🗑️ Clear Documents", use_container_width=True):
        vector_store.clear_saved()
        st.session_state.pipeline = None
        st.session_state.report = []
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()

    if st.sidebar.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def main():
    init_state()
    sidebar()

    st.title("📄 Domain-Specific RAG Chatbot")
    st.caption(
        "Answers come only from your uploaded PDFs. AI can make mistakes — "
        "please verify high-stakes information in the original document."
    )

    if st.session_state.pipeline is None:
        st.info("👈 Upload one or more PDFs in the sidebar and click **Process Documents** to begin.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources"))

    question = st.chat_input(
        "Ask a question about your documents...", disabled=st.session_state.pipeline is None
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = st.session_state.pipeline.answer(question)
            st.markdown(result["answer"])
            render_sources(result["sources"])
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )


main()
