import json
from pathlib import Path
from rag_common import keyword_overlap

LINKS_PATH = Path("data/normalized/links/jira_modules.jsonl")
CATALOG_PATH = Path("data/normalized/catalog.jsonl")

MODULE_PREFIXES = {
    "core": "core/",
    "sql": "sql/",
    "streaming": "streaming/",
    "python/pyspark": "python/pyspark/",
    "docs": "docs/",
}

CODE_EXTENSIONS = {".scala", ".java", ".py", ".md"}
MAX_RELATED_JIRA = 5
MAX_CODE_FILES = 5
MAX_CODE_CHARS = 2500

def load_jira_module_indexes():
    """Returns (ticket_to_moduels, module_to_tickets)."""
    ticket_to_modules: dict[str, set[str]] = {}
    module_to_tickets: dict[str, list[dict]] = {}
    for line in LINKS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("mapped") or not row.get("module"):
            continue
        key = row["jira_key"]
        module = row["module"]
        ticket_to_modules.setdefault(key, set()).add(module)
        bucket = module_to_tickets.setdefault(module, [])
        if not any(t["jira_key"] == key for t in bucket):
            bucket.append({
                "jira_key": key,
                "summary": row.get("summary") or "",
                "status": row.get("status") or "",
                "component": row.get("component") or "",
            })
    return ticket_to_modules, module_to_tickets

def load_catalog_by_module():
    """Returns module -> list of {path_or_key, text_path}."""
    catalog_by_module: dict[str, list[dict]] = {}
    for line in CATALOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source_type") != "code_file":
            continue
        path = row["path_or_key"]
        ext = Path(path).suffix.lower()
        if ext not in CODE_EXTENSIONS:
            continue
        if row.get("content_hash") == "skipped_large":
            continue
        module = None
        for mod, prefix in MODULE_PREFIXES.items():
            if path.startswith(prefix):
                module = mod
                break
        if not module:
            continue
        catalog_by_module.setdefault(module, []).append({
            "path_or_key": path,
            "text_path": row["text_path"],
        })

    return catalog_by_module
    
def read_code_snippet(text_path: str, max_chars: int = MAX_CODE_CHARS) -> str:
    path = Path(text_path)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated]"

def expand_context(seeds, query, by_key, ticket_to_modules, module_to_tickets, catalog_by_module):
    seen_keys = {s["path_or_key"] for s in seeds}
    seen_paths = set()
    related_candidates = []
    code_snippets = []
    for seed in seeds:
        key = seed["path_or_key"]
        if not key.startswith("SPARK-"):
            continue
        modules = ticket_to_modules.get(key, set())
        for module in modules:
            for ticket in module_to_tickets.get(module, []):
                tkey = ticket["jira_key"]
                if tkey in seen_keys:
                    continue
                if tkey in by_key:
                    text = by_key[tkey]["text"]
                else:
                    text = (
                        f"Summary: {ticket['summary']}\n"
                        f"Status: {ticket['status']}\n"
                        f"Component: {ticket['component']}\n"
                    )
                score = keyword_overlap(query, {"path_or_key": tkey, "text": text})
                related_candidates.append((score, module, tkey, text))
            added_for_module = 0
            for entry in catalog_by_module.get(module, []):
                if added_for_module >= MAX_CODE_FILES:
                    break
                path = entry["path_or_key"]
                if path in seen_paths:
                    continue
                snippet = read_code_snippet(entry["text_path"])
                if not snippet:
                    continue
                seen_paths.add(path)
                code_snippets.append({
                    "path": path,
                    "module": module,
                    "text": snippet,
                })
                added_for_module += 1
    related_candidates.sort(key=lambda x: x[0], reverse=True)
    related_jira = []
    for score, module, tkey, text in related_candidates:
        if tkey in seen_keys:
            continue
        seen_keys.add(tkey)
        related_jira.append({
            "jira_key": tkey,
            "module": module,
            "text": text,
        })
        if len(related_jira) >= MAX_RELATED_JIRA:
            break
    return related_jira, code_snippets