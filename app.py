import streamlit as st
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
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
    reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    for page in reader.pages:
        text += page.extract_text() or ""

    st.subheader("Extracted Text")
    st.text_area("PDF Context", text, height=300)

    # --- Step 3: Chunk the Text ---
    chunks = split_text_into_chunks(text)
    st.write(f"🔹 Number of chunks: {len(chunks)}")

    if st.checkbox("Show Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f'**Chunk {i+1}**')
            st.write(chunk)

    # --- Step 4: Generate Embeddings ---
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None

    if st.button('Generate Embeddings'):
        with st.spinner('Generating Embeddings...'):
            embeddings = get_embeddings(chunks)
            st.session_state.embeddings = embeddings
        st.success("Embeddings generated!")

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
