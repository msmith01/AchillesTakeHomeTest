import os
import glob
import re
import fitz
import tiktoken
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import chromadb
import ollama

RAW_DATA_DIR = "raw_data"
CHROMA_DIR   = "chroma_store"

# ── Ingest PDFs ───────────────────────────────────────────────────────────────

pdf_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pdf"))

documents = {}
for path in pdf_files:
    name = os.path.basename(path)
    with fitz.open(path) as doc:
        pages = [page.get_text() for page in doc]
        tagged = "\n".join(f"[PAGE {i+1}]\n{text}" for i, text in enumerate(pages))
    documents[name] = tagged
    print(f"Ingested: {name} ({len(pages)} pages)")

# ── End Ingest PDFs ───────────────────────────────────────────────────────────


# ── Recursive Token Chunking (10–15% sentence overlap) ───────────────────────

CHUNK_SIZE   = 500
OVERLAP_SIZE = 50
ENCODING     = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_into_units(text: str, level: str) -> list[str]:
    if level == "paragraph":
        return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if level == "sentence":
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if level == "word":
        return text.split()
    return list(text)


def recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    if count_tokens(text) <= chunk_size:
        return [text]

    for level in ("paragraph", "sentence", "word", "character"):
        units = split_into_units(text, level)
        if len(units) > 1:
            chunks = []
            current_tokens = []
            current_count  = 0

            for unit in units:
                unit_tokens = ENCODING.encode(unit + " ")
                if current_count + len(unit_tokens) > chunk_size and current_tokens:
                    chunks.append(ENCODING.decode(current_tokens).strip())
                    current_tokens = current_tokens[-OVERLAP_SIZE:]
                    current_count  = len(current_tokens)
                current_tokens += unit_tokens
                current_count  += len(unit_tokens)

            if current_tokens:
                chunks.append(ENCODING.decode(current_tokens).strip())

            final = []
            for chunk in chunks:
                if count_tokens(chunk) > chunk_size:
                    final.extend(recursive_chunk(chunk, chunk_size))
                else:
                    final.append(chunk)
            return final

    tokens = ENCODING.encode(text)
    return [
        ENCODING.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens), chunk_size - OVERLAP_SIZE)
    ]


def extract_start_page(chunk_text: str) -> int:
    match = re.search(r"\[PAGE (\d+)\]", chunk_text)
    return int(match.group(1)) if match else 1


chunked_documents = {}
for name, text in documents.items():
    chunks = recursive_chunk(text)
    chunked_documents[name] = chunks
    print(f"{name}: {len(chunks)} chunks")

# ── End Recursive Token Chunking ──────────────────────────────────────────────


# ── Embed Chunks — Qwen3-Embedding-8B (Alibaba) ───────────────────────────────

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_BATCH = 32
MAX_LENGTH  = 8192
device      = "cuda"

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="left")
embed_model     = AutoModel.from_pretrained(
    EMBED_MODEL,
    torch_dtype=torch.bfloat16,
).to(device)
embed_model.eval()


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def embed_texts(texts: list[str]) -> Tensor:
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        encoded = embed_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = embed_model(**encoded)
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)


embedded_documents = {}
for name, chunks in chunked_documents.items():
    embeddings = embed_texts(chunks)
    embedded_documents[name] = {"chunks": chunks, "embeddings": embeddings}
    print(f"{name}: embedded {embeddings.shape[0]} chunks → shape {tuple(embeddings.shape)}")

# ── End Embed Chunks ──────────────────────────────────────────────────────────


# ── Store in ChromaDB ─────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection    = chroma_client.get_or_create_collection(name="earnings_calls")

for name, data in embedded_documents.items():
    chunks     = data["chunks"]
    embeddings = data["embeddings"].tolist()
    source     = name.replace(".pdf", "")
    parts      = source.split("_")
    gvkey, year, transcript_id = parts[0], parts[1], parts[2]

    ids       = [f"{source}_chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source"       : name,
            "gvkey"        : gvkey,
            "year"         : year,
            "transcript_id": transcript_id,
            "chunk_index"  : i,
            "page"         : extract_start_page(chunks[i]),
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids        = ids,
        embeddings = embeddings,
        documents  = chunks,
        metadatas  = metadatas,
    )
    print(f"Stored {len(chunks)} chunks from {name} in ChromaDB")

print(f"Total documents in collection: {collection.count()}")

# ── End Store in ChromaDB ─────────────────────────────────────────────────────


# ── Retrieval — Top K ─────────────────────────────────────────────────────────

TOP_K = 5
RETRIEVAL_TASK = "Given a financial earnings call transcript, retrieve the most relevant passages that answer the query"


def get_detailed_instruct(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


def retrieve(query: str, k: int = TOP_K, year_filter: str = None) -> list[dict]:
    query_text      = get_detailed_instruct(RETRIEVAL_TASK, query)
    query_embedding = embed_texts([query_text]).tolist()[0]

    where = {"year": year_filter} if year_filter else None

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = k,
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "score"         : round(1 - dist, 4),
            "year"          : meta["year"],
            "transcript_id" : meta["transcript_id"],
            "chunk_index"   : meta["chunk_index"],
            "page"          : meta["page"],
            "text"          : doc,
        })
    return hits


def print_hits(hits: list[dict]):
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] Score: {hit['score']} | Year: {hit['year']} | "
              f"Transcript: {hit['transcript_id']} | Page: {hit['page']}")
        print(hit["text"][:400])
        print()

# ── End Retrieval — Top K ─────────────────────────────────────────────────────


# ── Generation — RAG Answer via Local LLM (Ollama) ───────────────────────────

LOCAL_MODEL   = "qwen2.5:14b"
SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer questions strictly using the provided earnings call transcript excerpts. "
    "For every claim, cite the source using the format [Year: YYYY | Transcript: ID | Page: N]. "
    "If the answer cannot be found in the excerpts, say so clearly."
)


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(
            f"[Excerpt {i} | Year: {hit['year']} | "
            f"Transcript: {hit['transcript_id']} | Page: {hit['page']}]\n{hit['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def generate(query: str, k: int = TOP_K, year_filter: str = None) -> str:
    hits    = retrieve(query, k=k, year_filter=year_filter)
    context = build_context(hits)

    user_message = (
        f"Earnings call excerpts:\n\n{context}\n\n"
        f"Question: {query}"
    )

    full_response = []
    stream = ollama.chat(
        model    = LOCAL_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        stream = True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response.append(token)

    print()
    return "".join(full_response)


if __name__ == "__main__":
    print("\n=== RAG Answer: revenue 2018 ===\n")
    generate("What was the revenue in 2018?", year_filter="2018")

    print("\n=== RAG Answer: risks and challenges ===\n")
    generate("Did the company mention any risks or challenges?")

# ── End Generation ────────────────────────────────────────────────────────────
