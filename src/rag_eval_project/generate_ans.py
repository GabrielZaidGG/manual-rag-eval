"""Generation stage: builds a grounded prompt from retrieved chunks, calls the
LLM, and wires retrieval + prompting + generation into one pipeline entry point.
"""
from rag_eval_project.retrieval import retrieve_chroma, client


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Build a grounded prompt from retrieved chunks, plus a parallel
    chunks_used list for traceability (kept out of the prompt itself)."""
    context = "\n\n".join(f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(chunks))

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above. If the answer isn't in the context, say so."
    )

    chunks_used = [
        {"position": i + 1, "source": chunk["source"], "chunk_index": chunk["chunk_index"], "score": chunk["score"]}
        for i, chunk in enumerate(chunks)
    ]

    return prompt, chunks_used


def generate_answer(prompt: str) -> str:
    """Call the LLM with a fully-built prompt and return the answer text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def answer_question(query: str, k: int = 3) -> dict:
    """Full pipeline: retrieve -> build prompt -> generate answer.

    Returns {"answer": str, "chunks_used": list[dict]} so callers (scripts,
    eval harnesses, a future UI) get both the answer and traceability info.
    """
    results = retrieve_chroma(query, k)
    prompt, chunks_used = build_prompt(query, results)
    answer = generate_answer(prompt)
    return {"answer": answer, "chunks_used": chunks_used}