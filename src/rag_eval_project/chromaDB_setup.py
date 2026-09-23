"""One-time (or re-runnable) script: load the corpus, chunk it, embed it,
and upsert everything into the persistent Chroma collection.

Safe to re-run — upsert with deterministic IDs (source::chunk_index) means
re-running this updates records in place rather than duplicating them.
"""
from dotenv import load_dotenv
load_dotenv()

import chromadb
from rag_eval_project.corpus import chunk_corpus
from rag_eval_project.batch_embed import embed_batch


def build_vector_store(data_dir: str, chunk_size: int, overlap: int, batch_size: int) -> chromadb.Collection:
    chunks = chunk_corpus(data_dir, chunk_size, overlap)
    embedded = embed_batch(chunks, batch_size=batch_size)

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="prompt_injection_papers",
        configuration={"hnsw": {"space": "cosine"}},
    )

    ids = [f"{chunk['source']}::{chunk['chunk_index']}" for chunk in embedded]
    documents = [chunk["text"] for chunk in embedded]
    embeddings = [chunk["embedding"] for chunk in embedded]
    metadatas = [{"source": chunk["source"], "chunk_index": chunk["chunk_index"]} for chunk in embedded]

    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return collection


if __name__ == "__main__":
    collection = build_vector_store("data", chunk_size=1000, overlap=200, batch_size=100)
    print(f"Collection ready — {collection.count()} chunks stored.")