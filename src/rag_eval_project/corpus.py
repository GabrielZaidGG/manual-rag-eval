from rag_eval_project.loader import load_pdf
from rag_eval_project.chunker import chunk_text
from pathlib import Path

def chunk_corpus(data_dir: str, chunk_size: int, overlap: int) -> list[dict]:
    directory = Path(data_dir)
    pdf_files = sorted(directory.glob("*.pdf"))

    chunked_corpus =[]

    for pdf_path in pdf_files:
        chunk_counter=0
        text = load_pdf(str(pdf_path))
        file_chunks = chunk_text(text,chunk_size,overlap)
        for chunk in file_chunks:
            chunked_corpus.append({"text": chunk, "source": str(pdf_path), "chunk_index": chunk_counter})
            chunk_counter +=1
    return chunked_corpus



