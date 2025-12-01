import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PERSIST_DIR = ".chroma"
COLLECTION = "knowmyrights"

class QASystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings())
        self.coll = self.client.get_collection(COLLECTION)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def ask(self, query: str, k: int = 5):
        # Retrieve top-k chunks and form a concise answer
        # For MVP we simply return the top chunks as "answer" snippets.
        results = self.coll.query(
            query_embeddings=self.model.encode([query], normalize_embeddings=True).tolist(),
            n_results=k,
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"text": d, "source": m.get("source", "")} for d, m in zip(docs, metas)]
