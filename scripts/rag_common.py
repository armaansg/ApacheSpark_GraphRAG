"""Shared RAG utilities: index loading, retrieval, API calls."""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data/normalized/chunks/chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/normalized/chunks/embeddings.jsonl"
API_KEY = os.environ["COMPANY_API_KEY"]
BASE = os.environ["COMPANY_API_BASE"].rstrip("/")
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
CHAT_MODEL = os.environ["CHAT_MODEL"]
TOP_K = 10
MAX_PRIMARY_SEEDS = 5
LEXICAL_WEIGHT = 0.12
JIRA_KEY_RE = re.compile(r"SPARK-\d+", re.I)
STOPWORDS = frozenset({
    "about", "any", "are", "can", "does", "for", "from", "have", "how", "issue",
    "issues", "jira", "spark", "tell", "that", "the", "there", "ticket", "tickets",
    "what", "when", "where", "which", "who", "why", "with", "would", "you",
})


def extract_jira_keys(query: str) -> list[str]:
    seen = set()
    keys = []
    for match in JIRA_KEY_RE.finditer(query):
        key = match.group(0).upper()
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def missing_jira_keys(keys: list[str], by_key: dict) -> list[str]:
    return [key for key in keys if key not in by_key]


def primary_seeds(hits: list, keys: list[str], by_key: dict, max_generic: int = MAX_PRIMARY_SEEDS) -> list:
    """Seeds used for graph expansion (subset of retrieve hits)."""
    if keys:
        pinned = [by_key[key] for key in keys if key in by_key]
        return pinned
    return hits[:max_generic]

def keyword_overlap(query: str, item: dict) -> float:
    words = [
        w for w in re.findall(r"[a-z0-9]{3,}", query.lower())
        if w not in STOPWORDS
    ]
    if not words:
        return 0.0
    haystack = f"{item['path_or_key']} {item.get('text', '')}".lower()
    return sum(1 for w in words if w in haystack) / len(words)

def post(path, payload):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("API error:", e.code, body[:500])
        raise

def embed_text(text: str):
    data = post("/embeddings", {"model": EMBED_MODEL, "input": text})
    return data["data"][0]["embedding"]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def load_index():
    texts = {}
    for line in CHUNKS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts[row["chunk_id"]] = row["text"]
    index = []
    by_key = {}
    for line in EMBEDDINGS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        chunk_id = row["chunk_id"]
        item = {
            "chunk_id": chunk_id,
            "doc_id": row["doc_id"],
            "source_type": row["source_type"],
            "path_or_key": row["path_or_key"],
            "embedding": row["embedding"],
            "text": texts[chunk_id],
        }
        index.append(item)
        by_key[item["path_or_key"]] = item
    return index, by_key

def retrieve(query: str, index, by_key, k=TOP_K):
    keys = extract_jira_keys(query)
    pinned = []
    used_ids = set()
    for key in keys:
        item = by_key.get(key)
        if item and item["chunk_id"] not in used_ids:
            pinned.append(item)
            used_ids.add(item["chunk_id"])
    qvec = embed_text(query)
    scored = []
    for item in index:
        if item["chunk_id"] in used_ids:
            continue
        vec_score = cosine(qvec, item["embedding"])
        lex_score = keyword_overlap(query, item)
        combined = vec_score + LEXICAL_WEIGHT * lex_score
        scored.append((combined, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    slots = max(0, k - len(pinned))
    hits = pinned + [item for _, item in scored[:slots]]
    return hits, keys

def hits_for_answer(hits: list, keys: list[str]) -> list:
    """When the user names a ticket, answer from that ticket only."""
    if keys:
        pinned = [h for h in hits if h["path_or_key"] in keys]
        if pinned:
            return pinned
    return hits

def chat_completion(prompt: str, max_tokens: int = 800) -> str:
    data = post("/chat/completions", {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    })
    return data["choices"][0]["message"]["content"]

def answer_plain(query: str, hits, keys):
    context_parts = []
    for i, hit in enumerate(hits_for_answer(hits, keys), 1):
        context_parts.append(
            f"[Source {i}: {hit['path_or_key']}]\n{hit['text']}"
        )
    context = "\n\n".join(context_parts)
    prompt = f"""You are a helpful assistant for Apache Spark Jira issues.
Answer the question using only the context below. Be accurate and concise.
If the context does not contain enough information, say so clearly.
When citing information, reference the relevant SPARK key or source label.
Context:
{context}
Question: {query}
"""
    return chat_completion(prompt)