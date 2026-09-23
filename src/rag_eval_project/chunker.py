def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Split text into overlapping fixed-size chunks using a sliding window."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    i = 0
    chunk_list = []
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunk_list.append(chunk)
        i += chunk_size - overlap
    return chunk_list
