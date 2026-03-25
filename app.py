import re
import json
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import chromadb
from ollama import Client as OllamaClient
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR     = "chroma_store"
EMBED_MODEL    = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LOCAL_MODEL    = "qwen2.5:14b"
TOP_K          = 5
RETRIEVAL_K    = 20
QA_LOG         = "qa_log.md"

RETRIEVAL_TASK = "Given a financial earnings call transcript, retrieve the most relevant passages that answer the query"

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer questions strictly using the provided earnings call transcript excerpts. "
    "For every claim, cite the source using the format [Year: YYYY | Transcript: ID | Page: N]. "
    "If the answer cannot be found in the excerpts, say so clearly."
)

JUDGE_PROMPT = """You are a strict evaluation judge for a RAG system.
Given a QUESTION, SOURCE PASSAGES, and a GENERATED ANSWER, score the faithfulness of the answer.
Faithfulness measures whether every claim is directly supported by the sources — not prior knowledge.
Respond with valid JSON only: {"faithfulness_score": <float 0.0-1.0>, "reason": "<one sentence>"}
Scoring: 1.0=fully supported, 0.75=mostly supported, 0.5=half supported, 0.25=mostly hallucinated, 0.0=ignores sources."""

CITATION_PATTERN = re.compile(
    r"\[Year:\s*\d{4}\s*\|\s*Transcript:\s*\d+\s*\|\s*Page:\s*\d+\]"
)

# ── Load models ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer  = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="left")
    embedder   = AutoModel.from_pretrained(EMBED_MODEL, torch_dtype=torch.bfloat16).to(device)
    embedder.eval()

    re_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    reranker     = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL, torch_dtype=torch.bfloat16
    ).to(device)
    reranker.eval()

    chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection("earnings_calls")
    ollama     = OllamaClient(host="http://127.0.0.1:11434")

    return tokenizer, embedder, re_tokenizer, reranker, collection, ollama, device


tokenizer, embedder, re_tokenizer, reranker, collection, ollama_client, device = load_models()

# ── Core functions ────────────────────────────────────────────────────────────

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def embed_query(text: str) -> list[float]:
    query_text = f"Instruct: {RETRIEVAL_TASK}\nQuery: {text}"
    encoded = tokenizer(
        [query_text], padding=True, truncation=True, max_length=8192, return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        outputs = embedder(**encoded)
    pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
    return F.normalize(pooled, p=2, dim=1).cpu().tolist()[0]


def retrieve(query: str, year_filter: str = None) -> list[dict]:
    where   = {"year": year_filter} if year_filter else None
    results = collection.query(
        query_embeddings = [embed_query(query)],
        n_results        = RETRIEVAL_K,
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )
    candidates = [
        {
            "bi_score"      : round(1 - dist, 4),
            "year"          : meta["year"],
            "transcript_id" : meta["transcript_id"],
            "page"          : meta.get("page", "—"),
            "text"          : doc,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    pairs   = [[query, c["text"]] for c in candidates]
    encoded = re_tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        scores = reranker(**encoded).logits.squeeze(-1).float().cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
    for c, s in zip(candidates, scores):
        c["score"] = round(float(s), 4)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:TOP_K]


def build_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Excerpt {i+1} | Year: {h['year']} | Transcript: {h['transcript_id']} | Page: {h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )


def generate(query: str, hits: list[dict]):
    user_message = f"Earnings call excerpts:\n\n{build_context(hits)}\n\nQuestion: {query}"
    stream = ollama_client.chat(
        model    = LOCAL_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        stream = True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


def judge_faithfulness(query: str, context: str, answer: str) -> dict:
    prompt = f"QUESTION: {query}\n\nSOURCE PASSAGES:\n{context}\n\nGENERATED ANSWER:\n{answer}"
    response = ollama_client.chat(
        model    = LOCAL_MODEL,
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        format = "json",
    )
    try:
        result = json.loads(response["message"]["content"])
        return {
            "score" : round(float(result.get("faithfulness_score", 0.0)), 2),
            "reason": result.get("reason", ""),
        }
    except Exception:
        return {"score": None, "reason": "Could not parse judge response."}


def count_citations(text: str) -> int:
    return len(CITATION_PATTERN.findall(text))


def save_to_log(question: str, answer: str, hits: list[dict], judge: dict):
    with open(QA_LOG, "a", encoding="utf-8") as f:
        f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Q:** {question}\n\n")
        f.write(f"**A:** {answer}\n\n")
        f.write(f"**Faithfulness:** {judge['score']} — {judge['reason']}\n\n")
        f.write("**Sources:**\n")
        for h in hits:
            f.write(f"- Year: {h['year']} | Transcript: {h['transcript_id']} | Page: {h['page']} | Score: {h['score']}\n")
        f.write("\n---\n\n")

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Earnings RAG", page_icon="📈", layout="wide")
st.title("📈 Earnings Call RAG")
st.caption(f"AAR Corp · {collection.count()} chunks · {LOCAL_MODEL}")

with st.sidebar:
    st.header("⚙️ Settings")

    year_options = ["All years", "2018", "2019", "2020", "2021", "2022"]
    year_filter  = st.selectbox("Filter by year", year_options)
    year_filter  = None if year_filter == "All years" else year_filter

    top_k = st.slider("Top-K passages", min_value=1, max_value=10, value=TOP_K)
    show_judge = st.toggle("LLM-as-a-Judge", value=True)
    show_passages = st.toggle("Show retrieved passages", value=True)

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. 🔍 Embed query — Qwen3-8B\n"
        "2. 📦 Retrieve top 20 — ChromaDB\n"
        "3. 🔀 Re-rank — BGE cross-encoder\n"
        "4. 💬 Generate — qwen2.5:14b (Ollama)\n"
        "5. 🧑‍⚖️ Judge — faithfulness score\n"
        "6. 💾 Save — qa_log.md"
    )
    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            m = msg["meta"]
            cols = st.columns(3)
            cols[0].metric("Citations", m["citations"])
            cols[1].metric("Faithfulness", m["faith_score"] if m["faith_score"] is not None else "—")
            cols[2].metric("Top retrieval score", m["top_score"])

question = st.chat_input("Ask a question about the earnings calls...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Retrieving & re-ranking passages..."):
        hits = retrieve(question, year_filter=year_filter)

    if show_passages:
        with st.expander(f"📄 Retrieved passages — top {top_k} after re-ranking", expanded=True):
            for i, h in enumerate(hits[:top_k], 1):
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.metric(f"#{i} score", h["score"])
                    st.caption(f"Bi-enc: {h['bi_score']}")
                with col2:
                    st.markdown(f"**Year:** {h['year']} · **Transcript:** {h['transcript_id']} · **Page:** {h['page']}")
                    st.caption(h["text"][:400] + "…")
                st.divider()

    with st.chat_message("assistant"):
        answer = st.write_stream(generate(question, hits[:top_k]))

    citations  = count_citations(answer)
    context    = build_context(hits[:top_k])

    if show_judge:
        with st.spinner("🧑‍⚖️ Judge evaluating faithfulness..."):
            judge = judge_faithfulness(question, context, answer)
    else:
        judge = {"score": None, "reason": "Judge disabled"}

    with st.chat_message("assistant"):
        cols = st.columns(3)
        cols[0].metric("Citations found", citations)
        faith_label = judge["score"] if judge["score"] is not None else "—"
        cols[1].metric("Faithfulness score", faith_label)
        cols[2].metric("Top retrieval score", hits[0]["score"])
        if judge["reason"]:
            st.caption(f"🧑‍⚖️ {judge['reason']}")

    st.session_state.history.append({
        "role"   : "assistant",
        "content": answer,
        "meta"   : {
            "citations"  : citations,
            "faith_score": judge["score"],
            "top_score"  : hits[0]["score"],
        }
    })

    save_to_log(question, answer, hits[:top_k], judge)
