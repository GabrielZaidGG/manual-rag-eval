from dotenv import load_dotenv
load_dotenv()

import chromadb
from rag_eval_project.corpus import chunk_corpus
from rag_eval_project.batch_embed import embed_batch

chunks = chunk_corpus("data", 1000, 200)
embedded = embed_batch(chunks, batch_size=100)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="prompt_injection_papers",
    configuration={"hnsw": {"space": "cosine"}},
)

ids = [f"{chunk['source']}::{chunk['chunk_index']}" for chunk in embedded]
documents = [chunk["text"] for chunk in embedded]
embeddings = [chunk["embedding"] for chunk in embedded]
metadatas = [{"source": chunk["source"], "chunk_index": chunk["chunk_index"]} for chunk in embedded]

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)
print(collection.count())