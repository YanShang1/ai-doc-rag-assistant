import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.config import AppConfig
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_chain import answer_question


load_dotenv()

st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Q&A Assistant")
st.caption("Upload a PDF and ask questions grounded in the document content.")

config = AppConfig.from_env()

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, config.chunk_size, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, config.chunk_overlap, 50)
    top_k = st.slider("Retrieved chunks", 1, 8, config.top_k, 1)

    st.divider()
    st.markdown("### Current Models")
    st.write(f"LLM: `{config.openai_model}`")
    st.write(f"Embedding: `{config.embedding_model}`")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process Document", type="primary"):
        if not config.openai_api_key:
            st.error("Missing OPENAI_API_KEY. Please create a .env file first.")
            st.stop()

        with st.spinner("Reading and indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)

            documents = load_pdf(tmp_path)
            chunks = split_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            vector_store = create_vector_store(
                chunks,
                embedding_model=config.embedding_model,
                collection_name="uploaded_pdf_collection",
            )

            st.session_state.vector_store = vector_store
            st.session_state.messages = []

        st.success(f"Document processed successfully. Created {len(chunks)} chunks.")

st.divider()

if st.session_state.vector_store is None:
    st.info("Upload and process a PDF first.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the uploaded document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                result = answer_question(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    openai_model=config.openai_model,
                    top_k=top_k,
                )

            st.markdown(result["answer"])

            with st.expander("Sources"):
                for i, source in enumerate(result["sources"], start=1):
                    st.markdown(f"**Source {i} — Page {source.get('page', 'Unknown')}**")
                    st.write(source["content"])

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
