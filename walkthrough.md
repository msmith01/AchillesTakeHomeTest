# Walkthrough

## `read_earnings.py` — Save Earnings Reports to PDF

Reads from a dataset of 100,000+ earnings call transcripts and filters to a single company (AAR Corp, `gvkey = 1004`). Each transcript is saved as an individual `.pdf` file into the `raw_data/` folder, named `{gvkey}_{year}_{transcriptId}.pdf`.

---

## `process_earnings.py` — Ingest & Chunk

### 1. Ingest PDFs
Reads every `.pdf` in `raw_data/` using **PyMuPDF**, extracting the full transcript text from each file.

### 2. Recursive Token Chunking with Overlap
Each transcript is split into token-bounded chunks using a hierarchical fallback strategy:

- **Paragraphs** → **Sentences** → **Words** → **Characters**

Splitting is done in token space (via `tiktoken`) rather than by character count, ensuring chunks respect the LLM's actual context window.

- **Chunk size**: 500 tokens
- **Overlap**: 50 tokens (~10–15%) — a sliding window carried forward between chunks to preserve cross-boundary context.

### 3. Embed Chunks — Qwen3-Embedding-8B (Alibaba)
Each chunk is embedded using **Qwen/Qwen3-Embedding-8B** via HuggingFace Transformers. The model runs on the RTX 5080 in `bfloat16` for memory efficiency. Embeddings are produced in batches of 32, pooled via last-token pooling, and L2-normalised so that dot-product equals cosine similarity.

### 4. Store in ChromaDB
Embeddings and their source chunks are persisted to a local **ChromaDB** collection (`chroma_store/`) so the pipeline does not need to re-embed on subsequent runs. Each chunk is stored with metadata — `gvkey`, `year`, `transcript_id`, `chunk_index`, and **`page`** — enabling filtered retrieval and cited responses later.

Page numbers are tracked by injecting `[PAGE N]` markers into the text during ingestion; each chunk retains the page number where its content begins.

### 5. Retrieval — Top K
On a natural language query, the query is embedded using the same Qwen3 model (with a task instruction prefix) and compared against all stored chunk embeddings via cosine similarity. The top `k=5` most relevant chunks are returned, ranked by score, with their source metadata (year, transcript, page).

### 6. Generation — Local LLM via Ollama
The retrieved chunks are assembled into a cited context block and passed to a **local LLM** (`qwen2.5:14b`) running via **Ollama**. The model streams its answer token-by-token and is instructed to cite every claim using the format `[Year: YYYY | Transcript: ID | Page: N]`, grounding all responses in the source passages.

---

## Run Output — AAR Corp (`gvkey = 1004`)

23 transcripts processed covering **2018–2022**. The model loaded across 4 checkpoint shards. Each transcript produced between 9–25 chunks of 4096-dimensional embeddings. All 380 chunks were successfully stored in ChromaDB.

| Year | Transcripts | Total Chunks |
|------|-------------|--------------|
| 2018 | 4           | 67           |
| 2019 | 4           | 71           |
| 2020 | 4           | 84           |
| 2021 | 6           | 90           |
| 2022 | 5           | 68           |
| **Total** | **23** | **380**  |

- **Embedding shape**: `(n_chunks, 4096)` — 4096 is the hidden dimension of Qwen3-Embedding-8B
- **Model load time**: near-instant (~0s across 4 shards at 116 it/s)

---

## `evaluate.py` — Interactive Query & Evaluation (+ Re-ranking & Faithfulness)

Run with:
```
python evaluate.py
```

### Interactive Mode
Type any natural language question to get a streamed, cited answer. Retrieved passages are shown in a table before the answer is generated.

### Evaluation Mode
Type `eval` at the prompt to run 10 predefined financial questions through the full RAG pipeline. Results are saved line-by-line to `eval.jsonl` with the following fields per response:

| Field | Description |
|---|---|
| `question` | The input question |
| `answer` | Full LLM response |
| `num_citations` | Count of valid citation patterns found |
| `has_valid_citation` | Boolean — did the response cite at least one source? |
| `top_hit_score` | Cosine similarity of the best retrieved chunk |
| `avg_hit_score` | Mean cosine similarity across top-k chunks |
| `sources` | List of retrieved chunks with year, transcript, page, score |

**Summary metrics printed after evaluation:**
- % of responses with valid citations
- Average citations per response
- Average retrieval score
- Average faithfulness score (LLM-as-a-Judge, 0.0–1.0)

Terminal output is formatted using **Rich** — tables for retrieved passages and evaluation results, a progress bar during evaluation, and colour-coded citation validity (✓/✗).

### Re-ranking — BGE Cross-Encoder

Before passing retrieved passages to the LLM, a **cross-encoder re-ranker** (`BAAI/bge-reranker-v2-m3`) scores each `(query, passage)` pair jointly. The pipeline fetches `k=20` candidates from ChromaDB via bi-encoder similarity, re-ranks them with the cross-encoder, and returns the top 5. This two-stage approach improves precision: the bi-encoder casts a wide net cheaply; the cross-encoder refines ordering with full attention over both texts.

### LLM-as-a-Judge — Faithfulness Scoring

Each generated answer is scored for **faithfulness** by a second LLM call using a structured judge prompt. The judge evaluates whether every claim in the answer is directly supported by the retrieved source passages (not hallucinated from prior knowledge) and returns a score from 0.0 to 1.0 plus a one-sentence reason. The score and reason are written to `eval.jsonl` alongside the retrieval metrics.

| Score | Meaning |
|-------|---------|
| 1.0 | Every claim explicitly supported by sources |
| 0.75 | Most claims supported, minor unsupported details |
| 0.5 | Roughly half the claims are supported |
| 0.25 | Few claims supported, mostly hallucinated |
| 0.0 | Answer ignores the sources entirely |

### Next Steps — Retrieval Quality Metrics

> **Next steps:** Create a labelled dataset (question → list of relevant chunk IDs) to enable retrieval-level evaluation using **MRR (Mean Reciprocal Rank)** and **Hit@k** metrics. This would allow direct measurement of how well the bi-encoder and re-ranker surface the correct passages, independent of generation quality, and support tuning of chunk size, overlap, and embedding model choice.

### Q&A Log — `qa_log.md`
Every interactive answer is automatically appended to `qa_log.md` with a timestamp, the question, the full answer, and the source passages used. The file persists across sessions — shut down and come back the next day and new answers are appended to the same file.

Example entry:
```markdown
## 2026-03-24 14:32

**Q:** What was revenue in 2020?

**A:** Based on the excerpts... [Year: 2020 | Transcript: 2164018 | Page: 2]

**Sources:**
- Year: 2020 | Transcript: 2164018 | Page: — | Score: 0.87

---
```

---

## `app.py` — Streamlit UI

Run with:
```
streamlit run app.py
```

A chat-style web interface over the full RAG pipeline.

**Features:**
- Chat input with streaming LLM responses
- Sidebar controls: year filter (2018–2022 or all), top-k slider, toggles for retrieved passages and LLM-as-a-Judge
- Expandable passage viewer showing bi-encoder score, cross-encoder score, year, transcript, page, and excerpt for each retrieved chunk
- Per-response metrics: citation count, faithfulness score, top retrieval score
- Conversation history persisted in session state; clear button in sidebar
- Every Q&A automatically appended to `qa_log.md`

**Pipeline shown in sidebar:**
1. Embed query — Qwen3-8B
2. Retrieve top 20 — ChromaDB
3. Re-rank — BGE cross-encoder
4. Generate — qwen2.5:14b (Ollama)
5. Judge — faithfulness score
6. Save — qa_log.md
