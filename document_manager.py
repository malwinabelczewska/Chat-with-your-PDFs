import sqlite3
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DocumentInfo:
    """Document metadata"""
    doc_id: str
    filename: str
    content_hash: str
    upload_date: str
    chunk_count: int
    file_size: int

class DocumentManager:
    """Manages document metadata and prevents duplicates"""

    def __init__(self, db_path: str = "./documents.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                upload_date TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                file_size INTEGER NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def generate_content_hash(self, content: str) -> str:
        """Generate SHA256 hash of document content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def generate_doc_id(self, filename: str, content_hash: str) -> str:
        """Generate unique document ID"""
        # Use first 8 chars of hash + sanitized filename
        sanitized_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        return f"{content_hash[:8]}_{sanitized_filename}"

    def document_exists(self, content_hash: str) -> Optional[DocumentInfo]:
        """Check if document with this content hash already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT doc_id, filename, content_hash, upload_date, chunk_count, file_size
            FROM documents WHERE content_hash = ?
        ''', (content_hash,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return DocumentInfo(*result)
        return None

    def add_document(self, filename: str, content: str, chunk_count: int) -> Tuple[str, bool]:
        """
        Add document to database

        Returns:
            (doc_id, is_new) - doc_id and whether this is a new document
        """
        content_hash = self.generate_content_hash(content)

        # Check if document already exists
        existing = self.document_exists(content_hash)
        if existing:
            return existing.doc_id, False

        # Add new document
        doc_id = self.generate_doc_id(filename, content_hash)
        upload_date = datetime.now().isoformat()
        file_size = len(content.encode('utf-8'))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO documents (doc_id, filename, content_hash, upload_date, chunk_count, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, filename, content_hash, upload_date, chunk_count, file_size))

            conn.commit()
            conn.close()
            return doc_id, True

        except sqlite3.IntegrityError:
            # Handle case where doc_id already exists (very unlikely)
            conn.close()
            doc_id = f"{content_hash[:12]}_{sanitized_filename}"
            return self.add_document(f"copy_{filename}", content, chunk_count)

    def list_documents(self) -> List[DocumentInfo]:
        """List all documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT doc_id, filename, content_hash, upload_date, chunk_count, file_size
            FROM documents ORDER BY upload_date DESC
        ''')

        results = cursor.fetchall()
        conn.close()

        return [DocumentInfo(*row) for row in results]

    def get_document(self, doc_id: str) -> Optional[DocumentInfo]:
        """Get document by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT doc_id, filename, content_hash, upload_date, chunk_count, file_size
            FROM documents WHERE doc_id = ?
        ''', (doc_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return DocumentInfo(*result)
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def clear_all(self):
        """Clear all document metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM documents')
        conn.commit()
        conn.close()