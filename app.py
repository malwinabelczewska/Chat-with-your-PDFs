import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import faiss
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

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

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
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text += page_text + "\n"

        pdf_document.close()

    except Exception as e:
        st.error(f"Failed to read the PDF file: {e}")
        st.stop()

    if not text.strip():
        st.error("No text could be extracted from the PDF. Please check if the PDF contains readable text.")
        st.stop()

    st.subheader("Extracted Text")
    st.text_area("PDF Context", text[:1000] + "..." if len(text) > 1000 else text, height=300)

    # --- Step 3: Chunk the Text ---
    chunks = split_text_into_chunks(text)
    st.write(f"🔹 Number of chunks: {len(chunks)}")

    if st.checkbox("Show Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f'**Chunk {i+1}**')
            st.write(chunk)

    # --- Step 4: Check for existing FAISS index ---
    index_loaded = False
    if os.path.exists("faiss_index.index") and os.path.exists("docs.pkl"):
        if st.button("Load Existing Index"):
            try:
                index, saved_chunks = load_index()
                st.session_state.index = index
                st.session_state.chunks = saved_chunks
                st.session_state.embeddings = None  # Using FAISS index instead
                st.success("✅ Loaded existing FAISS index!")
                index_loaded = True
            except Exception as e:
                st.error(f"Failed to load FAISS index: {e}")

    # --- Step 5: Generate Embeddings and Create Index ---
    if not index_loaded and st.button("Generate Embeddings & Create Index"):
        with st.spinner("Generating embeddings... This may take a moment."):
            try:
                # Generate embeddings
                embeddings = get_embeddings(chunks)
                st.session_state.embeddings = embeddings
                st.session_state.chunks = chunks

                # Build FAISS index
                dimension = len(embeddings[0])  # Should be 1536 for Ada-002
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)

                # Save index and chunks to disk
                save_index(index, chunks)
                st.session_state.index = index

                st.success("✅ Embeddings generated and FAISS index created!")

            except Exception as e:
                st.error(f"❌ Failed to generate embeddings: {e}")

    # --- Step 6: Ask Questions ---
    # Check if we have either embeddings or index ready
    ready_for_questions = (
            st.session_state.index is not None or
            st.session_state.embeddings is not None
    )

    if ready_for_questions and st.session_state.chunks is not None:
        st.subheader("Ask a question about the PDF")
        query = st.text_input("Enter your question:")

        if query:
            with st.spinner("Searching relevant content and generating answer..."):
                try:
                    # Use FAISS index if available, otherwise use embeddings
                    search_target = st.session_state.index if st.session_state.index is not None else st.session_state.embeddings

                    top_chunks = search_similar_chunks(
                        query,
                        st.session_state.chunks,
                        search_target
                    )

                    answer = answer_question_with_context(query, top_chunks)

                    st.subheader("Answer")
                    st.write(answer)

                    if st.checkbox("Show Retrieved Chunks"):
                        for i, chunk in enumerate(top_chunks):
                            st.markdown(f'**Matched Chunk {i+1}**')
                            st.write(chunk)

                except Exception as e:
                    st.error(f"Error processing your question: {e}")

    elif not ready_for_questions:
        st.info("👆 Please generate embeddings or load an existing index to start asking questions.")

    # Show current status
    st.sidebar.header("Status")
    st.sidebar.write(f"📄 Chunks: {'✅' if st.session_state.chunks is not None else '❌'}")
    st.sidebar.write(f"🧠 Embeddings: {'✅' if st.session_state.embeddings is not None else '❌'}")
    st.sidebar.write(f"🔍 FAISS Index: {'✅' if st.session_state.index is not None else '❌'}")