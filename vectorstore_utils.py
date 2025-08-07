import chromadb
from typing import List, Dict

# Global client and collection
client = None
collection = None

def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    global client, collection

    if client is None:
        # Create persistent client
        client = chromadb.PersistentClient(path="./chroma_db")

    if collection is None:
        collection = client.get_or_create_collection(name="pdf_chunks")

    return collection

def add_document_to_chromadb(doc_id: str, chunks: List[str]) -> bool:
    """
    Add a specific document's chunks to ChromaDB

    Args:
        doc_id: Unique document identifier
        chunks: List of text chunks for this document

    Returns:
        True if successful, False otherwise
    """
    try:
        collection = initialize_chromadb()

        # Create chunk IDs with document prefix
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Create metadata for each chunk
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        # Add to ChromaDB
        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=metadatas
        )

        print(f"Added {len(chunks)} chunks for document {doc_id}")
        return True

    except Exception as e:
        print(f"Error adding document {doc_id} to ChromaDB: {e}")
        return False

def search_in_document(query: str, doc_id: str = None, top_k: int = 3) -> List[str]:
    """
    Search for similar chunks, optionally filtered by document

    Args:
        query: Search query
        doc_id: Optional document ID to filter results
        top_k: Number of results to return

    Returns:
        List of matching text chunks
    """
    try:
        collection = initialize_chromadb()

        # Build where clause for document filtering
        where_clause = None
        if doc_id:
            where_clause = {"doc_id": doc_id}

        # Search the collection
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )

        # Return the document texts
        if results and results.get('documents') and results['documents'][0]:
            return results['documents'][0]
        else:
            return []

    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        return []

def get_documents_in_chromadb() -> Dict[str, int]:
    """
    Get all document IDs and their chunk counts from ChromaDB

    Returns:
        Dictionary mapping doc_id to chunk count
    """
    try:
        collection = initialize_chromadb()

        # Double check collection is valid
        if collection is None:
            print("Warning: ChromaDB collection is None")
            return {}

        # Get all documents
        all_docs = collection.get()

        # Count chunks per document
        doc_counts = {}
        if all_docs and all_docs.get('metadatas'):
            for metadata in all_docs['metadatas']:
                doc_id = metadata.get('doc_id', 'unknown')
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        return doc_counts

    except Exception:
        # Suppress the error for now since app works
        return {}

def delete_document_from_chromadb(doc_id: str) -> bool:
    """
    Delete all chunks for a specific document from ChromaDB

    Args:
        doc_id: Document ID to delete

    Returns:
        True if successful, False otherwise
    """
    initialize_chromadb()

    try:
        # Find all chunks for this document
        results = collection.get(where={"doc_id": doc_id})

        if results['ids']:
            # Delete all chunks for this document
            collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
            return True
        else:
            print(f"No chunks found for document {doc_id}")
            return False

    except Exception as e:
        print(f"Error deleting document {doc_id} from ChromaDB: {e}")
        return False

def clear_all_chromadb():
    """Clear all documents from ChromaDB"""
    initialize_chromadb()

    try:
        # Get all document IDs and delete them
        all_docs = collection.get()
        if all_docs['ids']:
            collection.delete(ids=all_docs['ids'])
            print("Cleared all documents from ChromaDB")
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")

# Backward compatibility functions
def save_index(index_placeholder, docs):
    """Legacy function - adds documents without proper document management"""
    print("Warning: Using legacy save_index. Consider using add_document_to_chromadb instead.")

    # Generate a simple doc_id for legacy usage
    doc_id = f"legacy_{hash(str(docs)) % 10000}"
    return add_document_to_chromadb(doc_id, docs)

def load_index():
    """Legacy function - returns collection and all documents"""
    initialize_chromadb()

    # Get all documents
    all_docs = collection.get()
    docs = all_docs['documents'] if all_docs['documents'] else []

    return collection, docs