# Earnings Call RAG Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-111111?logo=ollama&logoColor=white)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Local First](https://img.shields.io/badge/Privacy-Local_First-16a34a)](#models-all-local-no-api-keys-required)

A fully local retrieval-augmented generation pipeline over AAR Corp earnings-call transcripts from 2018–2022. It converts transcripts to PDFs, chunks and embeds them locally, retrieves and reranks relevant passages, and generates answers with source citations and an automated faithfulness score.

## Why this project

Financial transcripts are lengthy and difficult to search consistently. This project provides a reproducible research workflow that:

- keeps documents and model execution local;
- combines dense retrieval with cross-encoder reranking;
- returns cited answers rather than unsupported summaries;
- records retrieval and faithfulness metrics for evaluation;
- supports both an interactive CLI and a Streamlit interface.

## Pipeline

```text
PDFs → recursive token chunking → Qwen embeddings → ChromaDB
                                                        ↓
Query → retrieve top 20 → BGE reranker → select top 5
                                                        ↓
                 qwen2.5:14b via Ollama → cited answer
                                                        ↓
                              LLM-as-judge faithfulness
```

## Models

All models run locally; no hosted-model API key is required.

| Role | Model | Runtime |
|---|---|---|
| Embeddings | `Qwen/Qwen3-Embedding-8B` | Hugging Face Transformers |
| Reranking | `BAAI/bge-reranker-v2-m3` | Hugging Face Transformers |
| Generation and evaluation | `qwen2.5:14b` | Ollama |

The embedding and reranking stages use `bfloat16` GPU inference. Hardware requirements depend on model loading and available acceleration.

## Repository structure

```text
.
├── 01_save_earnings.py   # Convert filtered transcript data to PDFs
├── process_earnings.py   # Chunk, embed, and persist the ChromaDB index
├── evaluate.py           # Interactive query and benchmark CLI
├── app.py                # Streamlit chat interface
├── EDA.ipynb             # Exploratory data analysis
├── requirements.txt      # Python dependencies
├── walkthrough.md        # Detailed technical walkthrough
├── qa_log.md             # Runtime-generated question and answer log
└── eval.jsonl            # Runtime-generated evaluation records
```

`raw_data/` and `chroma_store/` are intentionally excluded and must be generated locally.

## Installation

### Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/)
- Sufficient local memory and GPU capacity for the selected models

### Setup

```bash
git clone https://github.com/msmith01/AchillesTakeHomeTest.git
cd AchillesTakeHomeTest
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and pull the generation model:

```bash
pip install -r requirements.txt
ollama pull qwen2.5:14b
```

## Build the vector store

Place earnings-call PDFs in `raw_data/`, then run:

```bash
python process_earnings.py
```

If starting from the source CSV, run `01_save_earnings.py` first. The resulting embeddings are persisted in `chroma_store/`.

## Usage

### Streamlit interface

```bash
streamlit run app.py
```

The interface includes year filtering, retrieved-passage inspection, citation counts, retrieval scores, and faithfulness scoring.

### Interactive CLI and evaluation

```bash
python evaluate.py
```

Commands:

- enter a question to receive a streamed, cited answer;
- enter `eval` to execute the benchmark and write `eval.jsonl`;
- enter `quit` to exit.

## Evaluation output

Each `eval.jsonl` record contains the input question, generated answer, citation validation, faithfulness assessment, retrieval scores, and retrieved source metadata.

| Field | Meaning |
|---|---|
| `num_citations` | Number of recognized source citations |
| `has_valid_citation` | Whether at least one citation matches the expected format |
| `faithfulness_score` | Judge score from 0.0 to 1.0 |
| `top_hit_score` | Best reranker score |
| `avg_hit_score` | Mean score across selected passages |
| `sources` | Retrieved chunks and metadata |

## Chunking strategy

The recursive splitter falls back through:

```text
paragraphs → sentences → words → characters
```

- Chunk size: 500 `cl100k_base` tokens
- Overlap: 50 tokens
- Source-page metadata: retained through injected `[PAGE N]` markers

## Dataset snapshot

The current corpus contains 23 AAR Corp (`gvkey = 1004`) transcripts and 380 chunks.

| Year | Transcripts | Chunks |
|---:|---:|---:|
| 2018 | 4 | 67 |
| 2019 | 4 | 71 |
| 2020 | 4 | 84 |
| 2021 | 6 | 90 |
| 2022 | 5 | 68 |
| **Total** | **23** | **380** |

## Roadmap

- Add labelled relevance judgments for MRR and Hit@k measurement.
- Support hosted vector stores for remote deployment.
- Add automated tests for ingestion, citation parsing, and evaluation output.

## Contributing

Issues and focused pull requests are welcome. For substantial changes, open an issue first to discuss the proposed design and evaluation criteria.

## License

No open-source license is currently included. Unless a license is added, the repository remains copyrighted and reuse is not automatically granted.

See [`walkthrough.md`](walkthrough.md) for a detailed implementation walkthrough.
