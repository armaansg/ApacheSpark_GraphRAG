import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = Path("data/normalized/chunks/chunks.jsonl")
OUT_PATH = Path("data/normalized/chunks/embeddings.jsonl")

API_KEY = os.environ["COMPANY_API_KEY"]
BASE = os.environ["COMPANY_API_BASE"].rstrip("/")
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]

BATCH_SIZE = 50
SLEEP_SECONDS = 0.25

MAX_EMBED_CHARS = 12000

def prepare_text(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_EMBED_CHARS:
        return text
    return text[:MAX_EMBED_CHARS] + "\n\n[truncated for embedding]"

def embed_batch(texts):
    payload = {"model": EMBED_MODEL, "input": texts}
    req = urllib.request.Request(
        f"{BASE}/embeddings",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("API error: e.code, body[:500]")
        raise

    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

def load_chunks(limit=None):
    chunks = []
    for line in CHUNKS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        chunks.append(json.loads(line))
        if limit and len(chunks) >= limit:
            break
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only embed first N chunks")
    args = parser.parse_args()

    chunks = load_chunks(limit=args.limit)
    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    
    with OUT_PATH.open("w") as out:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = []
            for c in batch:
                if len(c["text"]) > MAX_EMBED_CHARS:
                    print(f"  truncating {c['chunk_id']} ({len(c['text'])} chars)")
                texts.append(prepare_text(c["text"]))
            vectors = embed_batch(texts)

            for chunk, embedding in zip(batch, vectors):
                row = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source_type": chunk["source_type"],
                    "path_or_key": chunk["path_or_key"],
                    "embedding": embedding,
                }
                out.write(json.dumps(row) + "\n")
                written += 1
                print(f"  done {written}/{len(chunks)}")
            time.sleep(SLEEP_SECONDS)
    print(f"Wrote {written} embeddings to {OUT_PATH}")


if __name__ == "__main__":
    main()