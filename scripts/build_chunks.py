import json
from pathlib import Path

CATALOG = Path("data/normalized/catalog.jsonl")
OUT_DIR = Path("data/normalized/chunks")
OUT_PATH = OUT_DIR / "chunks.jsonl"
LINKS_PATH = Path("data/normalized/links/jira_modules.jsonl")

def load_components_by_key() -> dict[str, str]:
    by_key: dict[str, set[str]] = {}
    if not LINKS_PATH.exists():
        return {}
    for line in LINKS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        component = row.get("component")
        if component:
            by_key.setdefault(row["jira_key"], set()).add(component)
        return {key: ", ".join(sorted(comps)) for key, comps in by_key.items()}

def build_jira_text(row: dict, components_by_key: dict[str, str]) -> str:
    key = row["path_or_key"]
    meta = row.get("metadata") or {}
    summary = meta.get("summary") or ""
    status = meta.get("status") or ""
    body = row.get("text") or ""
    components = components_by_key.get(key, "")
    parts = [f"Key: {key}", f"Summary: {summary}", f"Status: {status}"]
    if components:
        parts.append(f"Components: {components}")
    parts.extend(["", body])
    return "\n".join(parts).strip()
    

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    components_by_key = load_components_by_key()
    written = 0
    skipped = 0
    with OUT_PATH.open("w") as out:
        for line in CATALOG.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_type") != "jira_issue":
                continue
            text = build_jira_text(row, components_by_key)
            if not text:
                skipped += 1
                continue
            key = row["path_or_key"]
            chunk = {
                "chunk_id": f"jira:{key}:0",
                "doc_id": row["doc_id"],
                "source_type": "jira_issue",
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