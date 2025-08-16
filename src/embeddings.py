from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_text(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of approx chunk_size tokens."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        start += chunk_size - chunk_overlap
    return chunks

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """Generate embeddings for list of text chunks."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

def create_faiss_index(embedding_dim):
    """Create a FAISS Flat index for L2 similarity search."""
    index = faiss.IndexFlatL2(embedding_dim)
    return index

def add_embeddings_to_index(index, embeddings):
    """Add embeddings to FAISS index."""
    index.add(embeddings)
