import hashlib, json
from pathlib import Path

ROOT = Path("data/raw/git/apache-spark")
REF = "v3.5.4-rc3"
OUT = Path("data/normalized/catalog.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

with OUT.open("w") as out:
    for line in Path("data/manifests/file_paths.txt").read_text().splitlines():
        p = line.strip()
        if not p:
            continue
        full = ROOT / p
        if not full.is_file():
            continue
        module = p.split("/")[0]
        if module == "python":
            module = "pyspark"
        row = {
            "doc_id": f"spark:{module}:{p}",
            "source_type": "code_file",
            "repo_id": f"spark:{module}",
            "path_or_key": p,
            "text_path": str(full),
            "metadata": {"ref": REF, "ext": full.suffix},
            "content_hash": file_hash(full) if full.stat().st_size < 500_000 else "skipped_large",
        }
        out.write(json.dumps(row) + "\n")
    for issue_file in sorted(Path("data/raw/jira/issues").glob("SPARK-*.json")):
        issue = json.loads(issue_file.read_text())
        f = issue["fields"]
        key = issue["key"]
        body = (f.get("description") or "") + "\n"
        comments = f.get("comment", {}).get("comments", [])
        for c in comments:
            body += f"\n--- comment ---\n{c.get('body','')}"
        row = {
            "doc_id": f"jira:{key}",
            "source_type": "jira_issue",
            "repo_id": "apache/spark",
            "path_or_key": key,
            "text": body[:50000],
            "metadata": {
                "summary": f.get("summary"),
                "status": (f.get("status") or {}).get("name"),
                "updated": f.get("updated"),
            },
        }
        out.write(json.dumps(row) + "\n")
print("Wrote", OUT)