# vector_store.py

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper for SentenceTransformers."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()
    

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()
    

def create_vector_store(documents: list[Document]):
    """Embed documents and store in FAISS vector DB."""
    embeddings = SentenceTransformerEmbeddings()
    return FAISS.from_documents(documents, embeddings)


def search_similar_chunks(vector_store: FAISS, query: str, k=4):
    """Search for top-k similar chunks for a given query."""
    return vector_store.similarity_search(query, k=k)
