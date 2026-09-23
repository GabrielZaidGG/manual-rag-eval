# RAG + Evaluation Pipeline (No Framework)

A Retrieval-Augmented Generation system built **from scratch in plain Python** — no LangChain, no high-level RAG framework — over a real corpus of research papers, with a lightweight evaluation layer to measure whether it actually works instead of assuming it does.

This project was built to understand the RAG mechanism end-to-end (retrieval math, prompt construction, generation, evaluation) before adopting any framework abstraction over it. A companion project reimplementing this same pipeline with LangChain is next, specifically so the trade-offs of the abstraction can be evaluated firsthand rather than taken on faith.

---

## What it does

Given a natural-language question, the system:

1. Embeds the question with the same model used to embed the corpus
2. Retrieves the top-k most semantically similar chunks from a persistent vector store
3. Constructs a grounded prompt from those chunks
4. Generates an answer constrained to only use the retrieved context — and explicitly refuses to answer when the context doesn't support it

```text
PDF
 ↓
Loader (pypdf)
 ↓
Chunker (fixed-size sliding window, with overlap)
 ↓
Chunks + Metadata (source file, chunk index)
 ↓
Embedding Model (OpenAI text-embedding-3-small)
 ↓
Vector Store (ChromaDB, persistent, cosine space)
 ↓
Retriever (manual cosine-similarity + Chroma-native, cross-verified)
 ↓
Prompt Builder (context + question + grounding instruction)
 ↓
LLM (OpenAI gpt-4o-mini)
 ↓
Answer + Traceability metadata
 ↓
Evaluation (13-query labeled set, automated pass/fail)
```

---

## Corpus

10 real, text-native research papers on **prompt injection** (implementation, effects, and mitigation) — sourced as arXiv-style PDFs, ~975K characters total. Chosen deliberately as topically related, overlapping-content documents rather than an "unrelated haystack," since semantically overlapping documents are a more honest stress test for retrieval quality than an easy needle-in-a-haystack setup.

---

## Key engineering decisions

Every major stage was implemented **manually first**, then cross-checked against the equivalent framework method, before either was trusted.

- **Manual retrieval before Chroma's `.query()`.** Cosine similarity was implemented by hand in NumPy — brute-force score-and-rank over all stored vectors — and verified to produce *identical* top-k results, order, and scores against Chroma's native HNSW-based query on this corpus. This confirmed the math was correct and made explicit that HNSW is an *approximate* nearest-neighbor method that only happened to be exact at this collection size (1,225 chunks).
- **Query and corpus embedded with the same explicit model call**, rather than relying on Chroma's default embedding function — avoiding a silent query/document embedding-space mismatch, which would produce meaningless similarity scores rather than merely worse ones.
- **Cosine similarity space set explicitly at collection creation.** ChromaDB defaults to L2 distance; cosine was configured deliberately to match the similarity metric used everywhere else in the project.
- **Deterministic, delimited chunk IDs** (`source::chunk_index`) for `upsert`, chosen over `uuid4()` specifically so re-running the pipeline updates records in place instead of duplicating them — verified by confirming `collection.count()` stays constant across repeated runs.
- **Source metadata kept out of the LLM-facing prompt but preserved in a parallel `chunks_used` structure** — the model doesn't need filenames to answer a question, but losing the link between an answer and the chunks that produced it would make debugging bad generations impossible after the fact.

Two real bugs were caught and root-caused during development, not hidden after the fact:
- A batching bug where a broken inner loop silently duplicated one chunk per batch instead of grouping distinct chunks — caught by manually tracing the loop on a small example before trusting it at scale.
- An ID-scheme migration issue where changing the ID format without clearing the existing vector store caused old and new records to coexist instead of overwriting — diagnosed by comparing stored IDs across runs, not guessed at.

---

## Evaluation

Evaluation is treated as part of the system, not an afterthought.

**Method:** a fixed, hand-labeled set of 13 queries, each tagged with an expected behavior:
- **In-domain** queries expected to produce a grounded, context-based answer (including several phrased with vague or unusual technical wording, deliberately chosen to stress-test retrieval)
- **Out-of-domain** queries — genuinely unrelated to the corpus — expected to produce an explicit refusal rather than a hallucinated answer

An automated runner calls the full pipeline for each query and checks the actual behavior against the expected label using a keyword-based refusal detector.

**Result:** 13/13 passed on the final run.

**A note on how that number was reached — deliberately left in, not cleaned up:**
Early runs scored 11/13 and 12/13, not because generation was wrong, but because the automated *grader* failed to recognize valid refusal phrasings it hadn't seen yet (e.g. "context provided does not contain" vs. "provided context does not contain"; "is not in the context"; "does not mention anything about"). Manual inspection confirmed generation was correct in every case — only the keyword-matching grader lagged behind the model's variable phrasing. The phrase list was expanded using only phrasing actually observed in real runs, not tuned defensively.

This is a known, explicitly accepted limitation: keyword matching is cheap and adequate for a first-pass eval layer, but temperature > 0 means the model can phrase a refusal in effectively unbounded ways, so a finite phrase list can never guarantee full coverage. Closing that gap properly would mean an LLM-as-judge grader — intentionally deferred to a follow-up project with a heavier evaluation framework, rather than solved by continuing to patch the phrase list.

---

## Known limitations

- **Chunking is naive fixed-size character splitting** with no sentence/paragraph awareness — chosen deliberately to learn the windowing/overlap mechanism explicitly before introducing NLP-aware splitting. Confirmed (via a live generation example) to occasionally produce whole-chunk bibliography fragments or mid-sentence cuts that retrieval can score highly despite being poor context.
- **No caching** — every pipeline run re-embeds all chunks and re-queries Chroma from scratch; there's no skip-if-unchanged logic.
- **No retry/error-handling** for transient API failures (embeddings or generation) mid-run.
- **Single-vendor generation** — originally designed as a deliberate multi-vendor split (OpenAI for embeddings, since Anthropic has no embeddings endpoint; Anthropic for generation), but generation currently also runs on OpenAI (`gpt-4o-mini`) due to available API credit. Documented as a deviation from the original design, not an oversight.
- **Keyword-based eval grader**, as described above — adequate for this project's scope, not a substitute for semantic evaluation.
- **Watermark/citation noise** in extracted text was investigated (see below) but not integrated into the live pipeline.

### Extraction cleaning — explored, decision made not to (yet) integrate

- A per-page arXiv watermark pattern was confirmed present once per document (10/10) and a regex fix was built, but not yet wired into the loader.
- Inline citation markers were investigated and their impact **tested empirically**, not assumed: the same sentence with and without its citation bracket produced 0.9649 cosine similarity, and the corpus was found to use two incompatible citation styles. Conclusion: stripping citations would add real implementation cost for negligible measured benefit — a deliberate decision **not** to build it, backed by evidence rather than intuition.
- Sentence-aware chunking was investigated as a fix for the mid-sentence-cut problem but deferred: `nltk.sent_tokenize` was found to mis-split domain-specific abbreviations common in this corpus (e.g. `"Fig. 3"`), and the severity of that gap wasn't yet measured against the benefit of leaving naive chunking in place.

---

## What I'd change at production scale

- Replace fixed-size chunking with sentence- or section-aware chunking once a tokenizer suited to technical/academic text is chosen
- Replace the brute-force retrieval fallback with Chroma's HNSW path exclusively past the current collection size, where exact brute-force scanning stops being viable
- Add embedding caching keyed on content hash, to avoid re-embedding unchanged chunks
- Add retry/backoff for embedding and generation API calls
- Replace the keyword-based eval grader with an LLM-as-judge for refusal/groundedness detection
- Add retrieval-level evaluation (are the *right* chunks in the top-k, independent of what the LLM does with them) rather than only end-to-end evaluation
- Reintroduce the original multi-vendor design (OpenAI embeddings, Anthropic generation) as a deliberate provider-swap comparison

---

## Stack

Python · `uv` · pypdf · OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`) · ChromaDB (persistent, cosine space) · NumPy

## Project structure

```text
rag-eval-project/
├── data/                          # corpus PDFs
├── src/
│   └── rag_eval_project/
│       ├── loader.py              # PDF text extraction
│       ├── chunker.py             # fixed-size windowing with overlap
│       ├── corpus.py              # corpus-wide load + chunk orchestration
│       ├── chromaDB_setup.py      # Sets up chroma DB Collection
│       ├── batch_embed.py         # cosine similarity, manual + Chroma-native retrieval
│       ├── generate_ans.py        # batched embedding calls
│       ├── retrieval.py           # prompt construction, generation, full pipeline
│       └── evalqueries.py         # labeled evaluation query set
├── eval_runner.py                 # automated evaluation harness
├── pyproject.toml
└── uv.lock
```

---

## Status

Core pipeline (load → chunk → embed → retrieve → generate) and a lightweight evaluation layer are implemented and verified end-to-end. Not yet started: automated unit tests (pytest), embedding caching, extraction cleaning integration, and the planned LangChain comparison implementation.