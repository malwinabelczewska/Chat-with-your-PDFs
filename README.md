# Chat with your PDFs 🤖📄

Upload multiple PDFs and ask questions about their content using OpenAI's GPT and ChromaDB for semantic search.

## Features ✨

- **Multi-document support** - Upload and manage multiple PDFs
- **Smart duplicate detection** - Automatically prevents re-uploading the same document
- **Document library** - Browse, select, and manage your uploaded PDFs
- **Flexible search** - Chat with specific documents or search across all documents
- **Persistent storage** - Your documents stay saved between sessions
- **Document metadata tracking** - View upload dates, chunk counts, and file sizes
- **Natural language Q&A** - Ask questions in plain English

## Technology Stack 🛠️

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB with persistent storage
- **Document Management**: SQLite for metadata tracking
- **Embeddings**: OpenAI text-embedding-ada-002
- **LLM**: OpenAI GPT-3.5-turbo
- **PDF Processing**: PyMuPDF (fitz)

## Installation 🚀

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage 📖

### Upload Documents
1. Use the file uploader to select a PDF
2. The app automatically detects if the document is already stored
3. Click "Save This PDF" to add new documents to your library

### Chat with Documents
1. Select a document from the sidebar library, or choose "Search all documents"
2. Type your question in the chat input
3. Get AI-powered answers with source citations

### Manage Your Library
- **View all documents**: See upload dates, chunk counts, and file sizes in the sidebar
- **Delete documents**: Remove individual PDFs or clear all data
- **Switch between documents**: Easily change which document you're chatting with

## File Structure 📁

```
├── app.py                 # Main Streamlit application
├── utils.py              # Text processing and OpenAI utilities
├── document_manager.py   # SQLite-based document metadata management
├── vectorstore_utils.py  # ChromaDB vector store operations
├── requirements.txt      # Python dependencies
├── .env                  # OpenAI API key (create this)
├── chroma_db/           # ChromaDB storage (auto-created)
└── documents.db         # Document metadata (auto-created)
```

## Configuration ⚙️

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `TOKENIZERS_PARALLELISM`: Set to "false" to suppress warnings (optional)

### Customization
You can modify these parameters in the code:
- **Chunk size**: Adjust `max_tokens` in `split_text_into_chunks()`
- **Search results**: Change `top_k` parameter in search functions
- **Model selection**: Update the model names in `utils.py`

## Data Storage 💾

- **Vector embeddings**: Stored in `./chroma_db/` directory
- **Document metadata**: Stored in `./documents.db` SQLite database
- **Persistent**: All data survives application restarts
- **Local**: Everything stays on your machine

## Troubleshooting 🔧

### Common Issues

**"No OpenAI API key found"**
- Make sure you have a `.env` file with `OPENAI_API_KEY=your-key`

**"Error getting ChromaDB documents"**
- These warnings are usually harmless and don't affect functionality
- Make sure the `chroma_db/` directory has write permissions

**"Document already exists"**
- The app detects duplicates by content hash
- This prevents accidentally uploading the same PDF multiple times

### Performance Tips
- Larger PDFs take longer to process (more chunks = more embeddings)
- First-time setup of ChromaDB may take a moment
- Search performance improves with more context in your questions

## Roadmap 🗺️

Planned improvements:
- [ ] Hybrid retrieval (dense + keyword search)
- [ ] Semantic re-ranking for better answer quality
- [ ] Support for additional document formats (Word, txt, etc.)
- [ ] Advanced metadata filtering and search
- [ ] Document versioning and update tracking
