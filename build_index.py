import os, glob, re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PERSIST_DIR = ".chroma"
COLLECTION = "knowmyrights"

def load_docs(content_dir="content"):
    docs = []
    for path in glob.glob(os.path.join(content_dir, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # simple chunking by headings/paragraphs
        chunks = re.split(r"\n{2,}", text)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 50:
                continue
            docs.append((f"{os.path.basename(path)}::{i}", chunk, path))
    return docs

def main():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=True))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    docs = load_docs()
    ids, texts, metas = [], [], []
    for doc_id, chunk, path in docs:
        ids.append(doc_id)
        texts.append(chunk)
        metas.append({"source": path})

    print(f"Indexing {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    print("✅ Built index at", PERSIST_DIR)

if __name__ == "__main__":
    main()
