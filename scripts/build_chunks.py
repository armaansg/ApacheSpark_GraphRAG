import json
from pathlib import Path

CATALOG = Path("data/normalized/catalog.jsonl")
OUT_DIR = Path("data/normalized/chunks")
OUT_PATH = OUT_DIR / "chunks.jsonl"

def build_jira_text(row: dict) -> str:
    meta = row.get("metadata") or {}
    summary = meta.get("summary") or ""
    status = meta.get("status") or ""
    body = row.get("text") or ""
    return f"Summary: {summary}\nStatus: {status}\n\n{body}".strip()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with OUT_PATH.open("w") as out:
        for line in CATALOG.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_type") != "jira_issue":
                continue
            text = build_jira_text(row)
            if not text:
                skipped += 1
                continue
            key = row["path_or_key"]
            chunk = {
                "chunk_id": f"jira:{key}:0",
                "doc_id": row["doc_id"],
                "source_type": "jira_issues",
                "path_or_key": key,
                "text": text,
            }
            out.write(json.dumps(chunk) + "\n")
            written += 1

    print(f"Jira chunks written:    {written}")
    print(f"Skipped empty:          {skipped}")
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()