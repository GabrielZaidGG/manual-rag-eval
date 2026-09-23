import chromadb
from rag_eval_project.retrieval import retrieve,retrieve_chroma,client


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']}"
        for i, chunk in enumerate(chunks)
    )

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


def generate_answer(prompt:str) -> str:
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],)
    return response.choices[0].message.content

def answer_question(query: str, k: int = 3) -> dict:
    # 1. retrieve chunks for query, using k
    results= retrieve_chroma(query,k)

    # 2. build the prompt from query + chunks
    prompt,chunks_used = build_prompt(query,results)
    # 3. generate the answer from the prompt
    answer=generate_answer(prompt)
    # 4. return something structured — think about what a caller
    #    (a script, an evaluation harness, a future UI) would need
    result_dict={"answer": answer,"chunks_used":chunks_used}
    return result_dict






