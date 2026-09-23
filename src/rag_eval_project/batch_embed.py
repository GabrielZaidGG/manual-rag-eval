from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI()

def embed_batch(chunks: list[dict], batch_size: int) -> list[dict]:
    embedding_result = []

    for i in range(0, len(chunks), batch_size):
        group = chunks[i:i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[chunk["text"] for chunk in group]
        )

        for chunk, embedding_obj in zip(group, response.data):
            chunk["embedding"] = embedding_obj.embedding
            embedding_result.append(chunk)

    return embedding_result