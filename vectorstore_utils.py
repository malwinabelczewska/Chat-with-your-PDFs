import faiss
import pickle

def save_index(index, docs, index_path="faiss_index.index", docs_path="docs.pkl"):
    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)

def load_index(index_path="faiss_index.index", docs_path="docs.pkl"):
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    return index, docs
