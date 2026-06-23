"""Graph expansion for Graph RAG: jira_modules + ranked catalog code snippets."""

import json
import re
from pathlib import Path

from rag_common import keyword_overlap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LINKS_PATH = PROJECT_ROOT / "data/normalized/links/jira_modules.jsonl"
CATALOG_PATH = PROJECT_ROOT / "data/normalized/catalog.jsonl"

MODULE_PREFIXES = {
    "core": "core/",
    "sql": "sql/",
    "streaming": "streaming/",
    "docs": "docs/",
    "python/pyspark": "python/pyspark/",
}

CODE_EXTENSIONS = {".scala", ".java", ".py", ".md"}
MAX_RELATED_JIRA = 3
MAX_CODE_FILES = 3
MAX_CODE_CHARS = 2500
MAX_TOTAL_EXPANDED_CHARS = 20000
CODE_PREVIEW_CHARS = 500

CODE_QUERY_TERMS = frozenset({
    "class", "code", "defined", "definition", "file", "files", "function",
    "implementation", "implement", "look", "method", "module", "path",
    "source", "where", "which",
})


def is_code_focused_query(query: str) -> bool:
    words = set(re.findall(r"[a-z]{3,}", query.lower()))
    return bool(words & CODE_QUERY_TERMS)


def expansion_limits(query: str, keys: list[str]) -> tuple[int, int]:
    """Return (max_related_jira, max_code_files) for this query."""
    if is_code_focused_query(query):
        return 1, MAX_CODE_FILES
    if keys:
        return 2, 2
    return MAX_RELATED_JIRA, 2


def seed_reference_text(seeds: list[dict]) -> str:
    return " ".join(seed.get("text", "")[:800] for seed in seeds)


def load_jira_module_indexes():
    """Returns (ticket_to_modules, module_to_tickets)."""
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


def rank_code_candidates(
    query: str,
    seed_text: str,
    entries: list[dict],
    seen_paths: set[str],
    limit: int,
) -> list[tuple[float, dict, str]]:
    candidates = []
    for entry in entries:
        path = entry["path_or_key"]
        if path in seen_paths:
            continue

        preview = read_code_snippet(entry["text_path"], max_chars=CODE_PREVIEW_CHARS)
        if not preview:
            continue

        rank_text = f"{path} {Path(path).name} {preview}"
        item = {"path_or_key": path, "text": rank_text}
        q_score = keyword_overlap(query, item)
        s_score = keyword_overlap(seed_text, item) if seed_text else 0.0
        score = 0.6 * q_score + 0.4 * s_score
        candidates.append((score, entry, preview))

    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[:limit]


def apply_expansion_cap(related_jira: list[dict], code_snippets: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Trim expanded context to MAX_TOTAL_EXPANDED_CHARS."""
    total = sum(len(r["text"]) for r in related_jira) + sum(len(c["text"]) for c in code_snippets)
    if total <= MAX_TOTAL_EXPANDED_CHARS:
        return related_jira, code_snippets, total

    trimmed_jira = list(related_jira)
    trimmed_code = list(code_snippets)

    while trimmed_jira or trimmed_code:
        total = sum(len(r["text"]) for r in trimmed_jira) + sum(len(c["text"]) for c in trimmed_code)
        if total <= MAX_TOTAL_EXPANDED_CHARS:
            break
        if trimmed_code and (not trimmed_jira or len(trimmed_code[-1]["text"]) >= len(trimmed_jira[-1]["text"])):
            trimmed_code.pop()
        elif trimmed_jira:
            trimmed_jira.pop()

    total = sum(len(r["text"]) for r in trimmed_jira) + sum(len(c["text"]) for c in trimmed_code)
    return trimmed_jira, trimmed_code, total


def expand_context(
    primary_seeds: list[dict],
    query: str,
    keys: list[str],
    by_key: dict,
    ticket_to_modules: dict,
    module_to_tickets: dict,
    catalog_by_module: dict,
):
    """
    Expand from primary seeds only (not all retrieve hits).

    Returns (related_jira, code_snippets, expanded_char_count).
    """
    if not primary_seeds:
        return [], [], 0

    max_related, max_code = expansion_limits(query, keys)
    seen_keys = {s["path_or_key"] for s in primary_seeds}
    seen_paths = set()
    seed_text = seed_reference_text(primary_seeds)
    related_candidates = []
    code_candidates = []

    modules_to_expand = set()
    for seed in primary_seeds:
        key = seed["path_or_key"]
        if key.startswith("SPARK-"):
            modules_to_expand |= ticket_to_modules.get(key, set())

    for module in modules_to_expand:
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
                    f"Component: {ticket['component']}"
                )

            item = {"path_or_key": tkey, "text": text}
            q_score = keyword_overlap(query, item)
            s_score = keyword_overlap(seed_text, item) if seed_text else 0.0
            score = 0.6 * q_score + 0.4 * s_score
            related_candidates.append((score, module, tkey, text))

        scan_limit = max(max_code * 40, 80)
        ranked_code = rank_code_candidates(
            query,
            seed_text,
            catalog_by_module.get(module, []),
            seen_paths,
            limit=scan_limit,
        )
        code_candidates.extend((score, module, entry) for score, entry, _ in ranked_code)

    related_candidates.sort(key=lambda row: row[0], reverse=True)
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
        if len(related_jira) >= max_related:
            break

    code_candidates.sort(key=lambda row: row[0], reverse=True)
    code_snippets = []
    for score, module, entry in code_candidates:
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
        if len(code_snippets) >= max_code:
            break

    related_jira, code_snippets, expanded_chars = apply_expansion_cap(related_jira, code_snippets)
    return related_jira, code_snippets, expanded_chars
