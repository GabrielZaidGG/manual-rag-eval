"""Retrieval stage: embed a query, score it against the stored corpus, and
return the top-k most similar chunks. Two implementations are provided —
retrieve() (manual, brute-force NumPy) and retrieve_chroma() (Chroma's
native HNSW query) — verified to produce identical results at this
collection size, so either can be trusted going forward.
"""
from openai import OpenAI
from dotenv import load_dotenv

import numpy as np
import chromadb

from rag_eval_project.batch_embed import embed_batch  # noqa: F401  (re-exported for callers of this module)

load_dotenv()
client_chroma = chromadb.PersistentClient(path="chroma_db")
collection = client_chroma.get_collection("prompt_injection_papers")
client = OpenAI()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors: (A·B) / (‖A‖ × ‖B‖)."""
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


def retrieve(query: str, k: int) -> list[dict]:
    """Manual retrieval: embed the query, score it against every stored chunk
    via brute-force cosine similarity in Python/numpy, and return the top-k
    chunks (each a dict with score, text, source, chunk_index, id).

    This is the from-scratch mechanism Chroma's collection.query() wraps.
    See retrieve_chroma() for the framework-native equivalent.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vec = np.array(response.data[0].embedding)

    result = collection.get(include=["embeddings", "documents", "metadatas"])

    pairs = []
    for embedding, document, metadata, chunk_id in zip(
        result["embeddings"], result["documents"], result["metadatas"], result["ids"]
    ):
        score = cosine_similarity(query_vec, np.array(embedding))
        pairs.append({
            "score": score,
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "id": chunk_id,
        })

    ranked = sorted(pairs, key=lambda pair: pair["score"], reverse=True)
    return ranked[:k]


def retrieve_chroma(query: str, k: int) -> list[dict]:
    """Framework-native retrieval: embed the query ourselves (so it's guaranteed
    to use the same model as the corpus), then let Chroma's HNSW index do the
    nearest-neighbor search instead of a brute-force Python scan.

    Returns the same shape as retrieve(), so the two are directly comparable.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vec = np.array(response.data[0].embedding)

    chroma_result = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for chunk_id, distance, document, metadata in zip(
        chroma_result["ids"][0],
        chroma_result["distances"][0],
        chroma_result["documents"][0],
        chroma_result["metadatas"][0],
    ):
        results.append({
            "score": 1 - distance,  # cosine space: distance = 1 - similarity
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "id": chunk_id,
        })

    return results