import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import os

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local imports
from utils import (
    split_text_into_chunks,
    answer_question_with_context
)
from vectorstore_utils import (
    add_document_to_chromadb,
    search_in_document,
    get_documents_in_chromadb,
    delete_document_from_chromadb,
    clear_all_chromadb
)
from document_manager import DocumentManager

load_dotenv()

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")
st.title("Chat with your PDFs 📄🤖")

# Initialize document manager
if 'doc_manager' not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if 'selected_doc_id' not in st.session_state:
    st.session_state.selected_doc_id = None

# --- Sidebar: Document Library ---
st.sidebar.header("📚 Document Library")

# Show existing documents
existing_docs = st.session_state.doc_manager.list_documents()
if existing_docs:
    st.sidebar.write(f"**{len(existing_docs)} documents stored:**")

    # Create a selectbox for choosing which document to chat with
    doc_options = {}
    for doc in existing_docs:
        display_name = f"{doc.filename} ({doc.chunk_count} chunks)"
        doc_options[display_name] = doc.doc_id

    # Add "All documents" option
    doc_options["🔍 Search all documents"] = "ALL"

    selected_display = st.sidebar.selectbox(
        "Chat with which document?",
        options=list(doc_options.keys()),
        key="doc_selector"
    )

    st.session_state.selected_doc_id = doc_options[selected_display]

    # Show document details
    if st.session_state.selected_doc_id != "ALL":
        selected_doc = st.session_state.doc_manager.get_document(st.session_state.selected_doc_id)
        if selected_doc:
            st.sidebar.write(f"📄 **{selected_doc.filename}**")
            st.sidebar.write(f"📅 Uploaded: {selected_doc.upload_date[:10]}")
            st.sidebar.write(f"📊 {selected_doc.chunk_count} chunks")
            st.sidebar.write(f"💾 {selected_doc.file_size:,} bytes")

            # Delete button for selected document
            if st.sidebar.button(f"🗑️ Delete {selected_doc.filename}"):
                if delete_document_from_chromadb(selected_doc.doc_id):
                    st.session_state.doc_manager.delete_document(selected_doc.doc_id)
                    st.sidebar.success("Document deleted!")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to delete document")

    # Clear all button
    if st.sidebar.button("🗑️ Clear All Documents"):
        clear_all_chromadb()
        st.session_state.doc_manager.clear_all()
        st.session_state.selected_doc_id = None
        st.sidebar.success("All documents cleared!")
        st.rerun()

else:
    st.sidebar.write("No documents stored yet.")
    st.sidebar.write("👆 Upload a PDF to get started!")

# --- Main Area: Upload and Chat ---

# --- Step 1: Upload PDF ---
st.header("📤 Upload New PDF")
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

    # --- Step 3: Check for duplicates ---
    content_hash = st.session_state.doc_manager.generate_content_hash(text)
    existing_doc = st.session_state.doc_manager.document_exists(content_hash)

    if existing_doc:
        st.info(f"📋 This PDF is already stored as: **{existing_doc.filename}**")
        st.info(f"Uploaded: {existing_doc.upload_date[:10]} | {existing_doc.chunk_count} chunks")

        if st.button("Use This Existing Document"):
            st.session_state.selected_doc_id = existing_doc.doc_id
            st.success(f"✅ Now chatting with: {existing_doc.filename}")
            st.rerun()
    else:
        # --- Step 4: Process new document ---
        st.subheader("📄 New Document Detected")

        # Show text preview
        with st.expander("Preview extracted text"):
            st.text_area("PDF Content", text[:1000] + "..." if len(text) > 1000 else text, height=200)

        # Chunk the text
        chunks = split_text_into_chunks(text)
        st.write(f"📊 This will create **{len(chunks)} chunks**")

        if st.checkbox("Show chunks preview"):
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                with st.expander(f"Chunk {i+1}"):
                    st.write(chunk)
            if len(chunks) > 3:
                st.write(f"... and {len(chunks) - 3} more chunks")

        # --- Step 5: Save new document ---
        if st.button("💾 Save This PDF"):
            with st.spinner("Processing and saving document..."):
                try:
                    # Add to document manager
                    doc_id, is_new = st.session_state.doc_manager.add_document(
                        uploaded_file.name, text, len(chunks)
                    )

                    if is_new:
                        # Add to ChromaDB
                        if add_document_to_chromadb(doc_id, chunks):
                            st.success(f"✅ Saved: {uploaded_file.name}")
                            st.success(f"📋 Document ID: {doc_id}")
                            st.session_state.selected_doc_id = doc_id
                            st.rerun()
                        else:
                            st.error("Failed to save to ChromaDB")
                    else:
                        st.info("Document already exists (this shouldn't happen)")

                except Exception as e:
                    st.error(f"❌ Failed to save document: {e}")

# --- Step 6: Chat Interface ---
if st.session_state.selected_doc_id:
    st.header("💬 Chat with Documents")

    if st.session_state.selected_doc_id == "ALL":
        st.info("🔍 Searching across all documents")
        search_scope = None
    else:
        selected_doc = st.session_state.doc_manager.get_document(st.session_state.selected_doc_id)
        if selected_doc:
            st.info(f"💬 Chatting with: **{selected_doc.filename}**")
            search_scope = st.session_state.selected_doc_id
        else:
            st.error("Selected document not found")
            st.stop()

    # Chat input
    query = st.text_input("Ask a question about your document(s):")

    if query:
        with st.spinner("Searching and generating answer..."):
            try:
                # Search for relevant chunks
                top_chunks = search_in_document(
                    query,
                    doc_id=search_scope,
                    top_k=5
                )

                if top_chunks:
                    # Generate answer
                    answer = answer_question_with_context(query, top_chunks)

                    st.subheader("💡 Answer")
                    st.write(answer)

                    # Show sources
                    with st.expander("📖 Source chunks"):
                        for i, chunk in enumerate(top_chunks):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(chunk)
                            st.markdown("---")
                else:
                    st.warning("No relevant information found for your question.")

            except Exception as e:
                st.error(f"Error processing your question: {e}")

elif existing_docs:
    st.info("👈 Select a document from the sidebar to start chatting!")
else:
    st.info("👆 Upload your first PDF to get started!")

# --- Status Footer ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    doc_count = len(existing_docs)
    st.metric("📚 Documents", doc_count)

with col2:
    chromadb_docs = get_documents_in_chromadb()
    chunk_count = sum(chromadb_docs.values())
    st.metric("📊 Total Chunks", chunk_count)

with col3:
    if st.session_state.selected_doc_id:
        if st.session_state.selected_doc_id == "ALL":
            st.metric("🎯 Search Scope", "All docs")
        else:
            selected_doc = st.session_state.doc_manager.get_document(st.session_state.selected_doc_id)
            if selected_doc:
                st.metric("🎯 Active Document", selected_doc.filename[:15] + "..." if len(selected_doc.filename) > 15 else selected_doc.filename)
    else:
        st.metric("🎯 Active Document", "None")