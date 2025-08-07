import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import faiss
import numpy as np
from vectorstore_utils import save_index, load_index
import os


# Local imports
from utils import (
    split_text_into_chunks,
    get_embeddings,
    search_similar_chunks,
    answer_question_with_context
)

load_dotenv()

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")
st.title("Chat with your PDFs 📄🤖")

# --- Step 1: Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # --- Step 2: Extract Text from PDF ---
    text = ""
    try:
        # Read the uploaded file
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text() + "\n"

        pdf_document.close()

    except Exception as e:
        st.error(f"Failed to read the PDF file: {e}")
        st.stop()

    st.subheader("Extracted Text")
    st.text_area("PDF Context", text, height=300)

    # --- Step 3: Chunk the Text ---
    chunks = split_text_into_chunks(text)
    st.write(f"🔹 Number of chunks: {len(chunks)}")

    if st.checkbox("Show Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f'**Chunk {i+1}**')
            st.write(chunk)

    if os.path.exists("faiss_index.index") and os.path.exists("docs.pkl"):
        try:
            index, chunks = load_index()
            st.session_state.index = index
            st.session_state.embeddings = None  # You no longer need manual embeddings
            st.success("Loaded existing FAISS index.")
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")

    # --- Step 4: Generate Embeddings ---
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None

    if st.button("Generate Embeddings"):
        with st.spinner("Generating Embeddings..."):
            embeddings = get_embeddings(chunks)
            st.session_state.embeddings = embeddings

            # Build and save FAISS index
            index = faiss.IndexFlatL2(1536)  # 1536 is the OpenAI embedding dimension
            index.add(embeddings)
            save_index(index, chunks)  # Save to disk

        st.success("Embeddings generated and index saved!")


    # --- Step 5: Ask a Question ---
    if st.session_state.embeddings:
        st.subheader("Ask a question about the PDF")
        query = st.text_input("Enter your question:")

        if query:
            with st.spinner("Searching relevant content and generating answer..."):
                top_chunks = search_similar_chunks(query, chunks, st.session_state.embeddings)
                answer = answer_question_with_context(query, top_chunks)

            st.subheader("Answer")
            st.write(answer)

            if st.checkbox("Show Retrieved Chunks"):
                for i, chunk in enumerate(top_chunks):
                    st.markdown(f'**Matched Chunk {i+1}**')
                    st.write(chunk)