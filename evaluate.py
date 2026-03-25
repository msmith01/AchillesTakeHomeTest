import re
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import chromadb
import ollama
from ollama import Client as OllamaClient
from transformers import AutoModelForSequenceClassification
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich import box

console = Console()

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
LOCAL_MODEL   = "qwen2.5:14b"
ollama_client = OllamaClient(host="http://127.0.0.1:11434")
TOP_K          = 5
RETRIEVAL_K    = 20       # fetch more candidates before re-ranking
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EVAL_FILE   = "eval.jsonl"
QA_LOG      = "qa_log.md"

RETRIEVAL_TASK = "Given a financial earnings call transcript, retrieve the most relevant passages that answer the query"

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer questions strictly using the provided earnings call transcript excerpts. "
    "For every claim, cite the source using the format [Year: YYYY | Transcript: ID | Page: N]. "
    "If the answer cannot be found in the excerpts, say so clearly."
)

CITATION_PATTERN = re.compile(
    r"\[Year:\s*\d{4}\s*\|\s*Transcript:\s*\d+\s*\|\s*Page:\s*\d+\]"
)

# ── Load embedding model ──────────────────────────────────────────────────────

device = "cuda"

with console.status("[bold cyan]Loading re-ranker model...", spinner="dots"):
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    reranker = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL, torch_dtype=torch.bfloat16
    ).to(device)
    reranker.eval()

console.print("[bold green]✓[/] Re-ranker loaded")

with console.status("[bold cyan]Loading embedding model...", spinner="dots"):
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="left")
    embed_model     = AutoModel.from_pretrained(
        EMBED_MODEL,
        torch_dtype=torch.bfloat16,
    ).to(device)
    embed_model.eval()

console.print("[bold green]✓[/] Embedding model loaded")

# ── Connect to ChromaDB ───────────────────────────────────────────────────────

with console.status("[bold cyan]Connecting to ChromaDB...", spinner="dots"):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection    = chroma_client.get_collection("earnings_calls")

console.print(f"[bold green]✓[/] ChromaDB connected — [bold]{collection.count()}[/] chunks\n")

# ── Embedding & Retrieval ─────────────────────────────────────────────────────

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def embed_query(text: str) -> list[float]:
    query_text = f"Instruct: {RETRIEVAL_TASK}\nQuery: {text}"
    encoded = embed_tokenizer(
        [query_text], padding=True, truncation=True, max_length=8192, return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        outputs = embed_model(**encoded)
    pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu().tolist()[0]


def retrieve(query: str, k: int = TOP_K, year_filter: str = None) -> list[dict]:
    where   = {"year": year_filter} if year_filter else None

    results = collection.query(
        query_embeddings = [embed_query(query)],
        n_results        = RETRIEVAL_K,
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "bi_encoder_score" : round(1 - dist, 4),
            "year"             : meta["year"],
            "transcript_id"    : meta["transcript_id"],
            "page"             : meta.get("page", "—"),
            "text"             : doc,
        })

    pairs   = [[query, c["text"]] for c in candidates]
    encoded = reranker_tokenizer(
        pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        rerank_scores = reranker(**encoded).logits.squeeze(-1).float().cpu().tolist()
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]
    for candidate, score in zip(candidates, rerank_scores):
        candidate["score"] = round(float(score), 4)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]


def build_context(hits: list[dict]) -> str:
    blocks = [
        f"[Excerpt {i+1} | Year: {h['year']} | Transcript: {h['transcript_id']} | Page: {h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    ]
    return "\n\n---\n\n".join(blocks)


def print_hits(hits: list[dict]):
    table = Table(title="Retrieved Passages", box=box.ROUNDED, show_lines=True)
    table.add_column("#",             style="bold cyan",  width=3)
    table.add_column("Score",         style="bold green", width=7)
    table.add_column("Year",          style="yellow",     width=6)
    table.add_column("Transcript",    style="magenta",    width=12)
    table.add_column("Page",          style="cyan",       width=6)
    table.add_column("Excerpt",       style="white",      no_wrap=False)
    for i, h in enumerate(hits, 1):
        table.add_row(
            str(i),
            str(h["score"]),
            h["year"],
            h["transcript_id"],
            str(h["page"]),
            h["text"][:200] + "…",
        )
    console.print(table)

# ── Generation ────────────────────────────────────────────────────────────────

def generate(query: str, k: int = TOP_K, year_filter: str = None, silent: bool = False) -> tuple[str, list[dict]]:
    hits    = retrieve(query, k=k, year_filter=year_filter)
    context = build_context(hits)

    if not silent:
        print_hits(hits)
        console.print(Panel("[bold cyan]Generating answer...[/]", expand=False))

    user_message = f"Earnings call excerpts:\n\n{context}\n\nQuestion: {query}"

    tokens = []
    stream = ollama_client.chat(
        model    = LOCAL_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        stream = True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if not silent:
            console.print(token, end="")
        tokens.append(token)

    if not silent:
        console.print()

    answer = "".join(tokens)

    if not silent:
        from datetime import datetime
        with open(QA_LOG, "a", encoding="utf-8") as f:
            f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Q:** {query}\n\n")
            f.write(f"**A:** {answer}\n\n")
            f.write("**Sources:**\n")
            for h in hits:
                f.write(f"- Year: {h['year']} | Transcript: {h['transcript_id']} | Page: {h['page']} | Score: {h['score']}\n")
            f.write("\n---\n\n")

    return answer, hits

# ── Evaluation ────────────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    "What was the total revenue in 2018?",
    "Did the company mention any risks or challenges in 2019?",
    "What were the key operational highlights in 2020?",
    "How did the COVID-19 pandemic affect the business?",
    "What growth strategies did management discuss in 2021?",
    "Were there any acquisitions or partnerships mentioned?",
    "What was said about margins and profitability in 2022?",
    "How did the company perform in the aviation services segment?",
    "What forward-looking guidance was provided?",
    "Were there any mentions of supply chain issues?",
]


def has_valid_citation(text: str) -> bool:
    return bool(CITATION_PATTERN.search(text))


def count_citations(text: str) -> int:
    return len(CITATION_PATTERN.findall(text))


# ── LLM-as-a-Judge — Faithfulness scoring ────────────────────────────────────

JUDGE_PROMPT = """You are a strict evaluation judge for a RAG system.

Given a QUESTION, a set of SOURCE PASSAGES retrieved from earnings call transcripts, and a GENERATED ANSWER, your task is to score the faithfulness of the answer.

Faithfulness measures whether every claim in the answer is directly supported by the source passages — not by prior knowledge.

Respond with valid JSON only, in this exact format:
{{"faithfulness_score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Scoring guide:
- 1.0 — every claim is explicitly supported by the sources
- 0.75 — most claims supported, minor unsupported details
- 0.5 — roughly half the claims are supported
- 0.25 — few claims supported, mostly hallucinated
- 0.0 — answer ignores the sources entirely"""


def judge_faithfulness(question: str, context: str, answer: str) -> dict:
    prompt = (
        f"QUESTION: {question}\n\n"
        f"SOURCE PASSAGES:\n{context}\n\n"
        f"GENERATED ANSWER:\n{answer}"
    )
    response = ollama_client.chat(
        model    = LOCAL_MODEL,
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        format   = "json",
    )
    try:
        result = json.loads(response["message"]["content"])
        return {
            "faithfulness_score": float(result.get("faithfulness_score", 0.0)),
            "faithfulness_reason": result.get("reason", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"faithfulness_score": None, "faithfulness_reason": "parse error"}

# ── End LLM-as-a-Judge ────────────────────────────────────────────────────────


def run_evaluation():
    console.rule("[bold yellow]RAG Evaluation")
    records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold cyan]{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating questions...", total=len(EVAL_QUESTIONS))

        for question in EVAL_QUESTIONS:
            progress.update(task, description=f"[cyan]{question[:60]}…")
            answer, hits   = generate(question, silent=True)
            context        = build_context(hits)
            faithfulness   = judge_faithfulness(question, context, answer)

            records.append({
                "question"           : question,
                "answer"             : answer,
                "num_citations"      : count_citations(answer),
                "has_valid_citation" : has_valid_citation(answer),
                "faithfulness_score" : faithfulness["faithfulness_score"],
                "faithfulness_reason": faithfulness["faithfulness_reason"],
                "top_hit_score"      : hits[0]["score"] if hits else None,
                "avg_hit_score"      : round(sum(h["score"] for h in hits) / len(hits), 4) if hits else None,
                "sources"            : [
                    {"year": h["year"], "transcript_id": h["transcript_id"], "page": h["page"], "score": h["score"]}
                    for h in hits
                ],
            })
            progress.advance(task)

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total            = len(records)
    pct_cited        = round(sum(r["has_valid_citation"] for r in records) / total * 100, 1)
    avg_citations    = round(sum(r["num_citations"] for r in records) / total, 2)
    avg_score        = round(sum(r["avg_hit_score"] for r in records if r["avg_hit_score"]) / total, 4)
    faith_scores     = [r["faithfulness_score"] for r in records if r["faithfulness_score"] is not None]
    avg_faithfulness = round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None

    results_table = Table(title="Evaluation Results", box=box.ROUNDED)
    results_table.add_column("Metric",  style="bold cyan")
    results_table.add_column("Value",   style="bold green")
    results_table.add_row("Questions evaluated",        str(total))
    results_table.add_row("% with valid citations",     f"{pct_cited}%")
    results_table.add_row("Avg citations per response", str(avg_citations))
    results_table.add_row("Avg retrieval score",        str(avg_score))
    results_table.add_row("Avg faithfulness score",     str(avg_faithfulness))
    results_table.add_row("Results saved to",           EVAL_FILE)
    console.print(results_table)

    per_q_table = Table(title="Per-Question Summary", box=box.SIMPLE)
    per_q_table.add_column("Question",      style="white",       no_wrap=False)
    per_q_table.add_column("Citations",     style="cyan",        width=10)
    per_q_table.add_column("Valid",         style="bold",        width=7)
    per_q_table.add_column("Faithfulness",  style="magenta",     width=13)
    per_q_table.add_column("Top Score",     style="green",       width=10)
    for r in records:
        valid_str = "[green]✓[/]" if r["has_valid_citation"] else "[red]✗[/]"
        faith     = r["faithfulness_score"]
        if faith is None:
            faith_str = "[dim]—[/]"
        elif faith >= 0.75:
            faith_str = f"[green]{faith}[/]"
        elif faith >= 0.5:
            faith_str = f"[yellow]{faith}[/]"
        else:
            faith_str = f"[red]{faith}[/]"
        per_q_table.add_row(
            r["question"],
            str(r["num_citations"]),
            valid_str,
            faith_str,
            str(r["top_hit_score"]),
        )
    console.print(per_q_table)

# ── Interactive query loop ────────────────────────────────────────────────────

def interactive():
    console.print(Panel(
        "[bold cyan]RAG Query Interface[/]\n"
        "Type your question and press Enter\n"
        "[dim]Commands: [bold]eval[/] — run evaluation | [bold]quit[/] — exit[/]",
        box=box.ROUNDED,
    ))

    while True:
        question = console.input("\n[bold yellow]Question:[/] ").strip()
        if question.lower() == "quit":
            console.print("[bold red]Goodbye![/]")
            break
        if question.lower() == "eval":
            run_evaluation()
            continue
        if not question:
            continue
        console.rule()
        generate(question)
        console.rule()


if __name__ == "__main__":
    interactive()
