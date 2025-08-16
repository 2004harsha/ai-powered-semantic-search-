import os
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import streamlit as st
import os
from src.document_processor import extract_text
from src.embeddings import chunk_text, Embedder, create_faiss_index, add_embeddings_to_index
from src.llm_handler import QAPipeline
import tempfile

st.set_page_config(page_title="AI-powered PDF & Docs Semantic Search", layout="wide")

# Cache Embedder and QAPipeline for performance
@st.cache_resource
def load_embedder():
    return Embedder()

@st.cache_resource
def load_qa_pipeline():
    return QAPipeline()

def main():
    st.title("AI-powered PDF + Google Docs Semantic Search & Q&A")
    st.write("Upload your PDF or DOCX document and ask questions about its content.")

    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            temp_filepath = tmp.name

        # Extract text from uploaded file
        st.info("Extracting text from document...")
        try:
            raw_text = extract_text(temp_filepath)
            if not raw_text.strip():
                st.warning("No text found in the document.")
                return
            st.success("Text extracted successfully.")

            # Chunk and embed
            st.info("Processing document for semantic search...")
            embedder = load_embedder()
            chunks = chunk_text(raw_text)
            embeddings = embedder.embed_texts(chunks)

            # Create FAISS index
            index = create_faiss_index(embeddings.shape[1])
            add_embeddings_to_index(index, embeddings)

            # Prepare QA pipeline
            qa_pipeline = load_qa_pipeline()

            # User question input
            question = st.text_input("Ask a question about the document:")

            if question:
                # Embed query
                query_emb = embedder.embed_texts([question])
                k = 5  # Number of top chunks to retrieve

                # Search FAISS for similar chunks
                distances, indices = index.search(query_emb, k)
                retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]

                # Generate answer with LLM
                with st.spinner("Generating answer..."):
                    answer = qa_pipeline.get_answer(retrieved_chunks, question)

                st.markdown("### Answer:")
                st.write(answer)

        finally:
            # Clean temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

if __name__ == "__main__":
    main()
