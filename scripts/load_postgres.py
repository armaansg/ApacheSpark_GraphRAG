#!/usr/bin/env python3
"""Phase 2 ETL loader: JSONL -> Postgres 16 + pgvector (schema.sql from Phase 1).

Loads the six normalized data targets into the Graph-RAG store:

    catalog.jsonl        -> documents
    nodes.jsonl          -> nodes
    edges.jsonl          -> edges
    jira_modules.jsonl   -> jira_module_links
    chunks.jsonl         -> chunks          (embedding left NULL; see backfill_embeddings.py)
    (synthesized)        -> repos           (single 'apache/spark' row)

This is the ETL layer ONLY. It does not touch query-side code and it never
calls an embedding API — chunks.embedding stays NULL and is filled later by the
standalone scripts/backfill_embeddings.py.

--------------------------------------------------------------------------------
Connection / RLS
--------------------------------------------------------------------------------
Every table has RLS ENABLED and FORCED with a fail-closed policy comparing
org_id to the per-connection GUC `app.org_id`. If the GUC is unset,
current_setting('app.org_id', true) returns NULL and every row is denied
(both reads and writes). Therefore this loader runs

    SET app.org_id = 'apache'

immediately after connecting (and commits it, so it persists for the whole
session), and stamps org_id='apache' on every inserted row. Without both, inserts
are silently rejected by the WITH CHECK clause.

--------------------------------------------------------------------------------
Locked data-model decisions
--------------------------------------------------------------------------------
* apache/spark is ONE repo:  repo_id = 'apache/spark' on every row.
* org_id = 'apache' (constant) on every row.
* `module` is a COLUMN in {core, sql, streaming, pyspark, docs}, DERIVED from
  each record's file path / source. See `normalize_module` / `module_from_path`
  for the exact rules, reproduced here:

  path first segment  -> module        (documents from path_or_key, nodes from `file`)
      core            -> core
      sql             -> sql
      streaming       -> streaming
      docs            -> docs
      python          -> pyspark        # the python/pyspark source tree

  legacy repo_id      -> module         (catalog code_file rows carry spark:<module>)
      spark:core      -> core
      spark:sql       -> sql
      spark:streaming -> streaming
      spark:docs      -> docs
      spark:pyspark   -> pyspark

  jira module value   -> module         (jira_modules.jsonl `module` field)
      python/pyspark  -> pyspark        # reconcile: same as a python/... path
      core|sql|streaming|docs|pyspark -> itself
      null / unmapped -> NULL

  jira_issue documents (path_or_key = 'SPARK-nnnnn') have no path module segment,
  so their documents.module is NULL; the authoritative Jira->module mapping lives
  in jira_module_links (a Jira issue may map to several modules).

--------------------------------------------------------------------------------
Idempotency
--------------------------------------------------------------------------------
* Tables with a natural key (repos, documents, nodes, chunks) use
  INSERT ... ON CONFLICT (<pk>) DO UPDATE. Re-running updates in place.
  The chunks upsert deliberately does NOT overwrite embedding / embedding_version,
  so a later embedding backfill survives a loader re-run.
* Tables with NO unique key (edges, jira_module_links — both allow legitimate
  duplicate rows) are reloaded with DELETE (this org+repo) then INSERT inside one
  transaction. This is idempotent: row counts are identical after a re-run.

--------------------------------------------------------------------------------
Run order & usage
--------------------------------------------------------------------------------
    # schema.sql must already be applied to the target DB (Phase 1).
    export DATABASE_URL=postgresql://spark:spark@localhost:5433/sparkctx   # optional; this is the default
    python scripts/load_postgres.py                # load everything
    python scripts/load_postgres.py --only nodes edges   # subset

Load order is repos -> documents -> nodes -> edges -> jira_module_links ->
chunks. nodes are loaded before edges because edge.is_resolved is computed
against the set of node ids read from nodes.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "normalized"

CATALOG_PATH = DATA / "catalog.jsonl"
NODES_PATH = DATA / "graph" / "nodes.jsonl"
EDGES_PATH = DATA / "graph" / "edges.jsonl"
JIRA_MODULES_PATH = DATA / "links" / "jira_modules.jsonl"
CHUNKS_PATH = DATA / "chunks" / "chunks.jsonl"

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://spark:spark@localhost:5433/sparkctx"
)

# Locked constants (single-repo, single-org world).
ORG_ID = os.environ.get("SPARK_ORG_ID", "apache")
REPO_ID = os.environ.get("SPARK_REPO_ID", "apache/spark")
REPO_NAME = os.environ.get("SPARK_REPO_NAME", "Apache Spark")
# Every catalog code_file carries metadata.ref = v3.5.4-rc3; use it as the pin.
INDEXED_SHA = os.environ.get("SPARK_INDEXED_SHA", "v3.5.4-rc3")

BATCH = 5000

# ---------------------------------------------------------------------------
# Module derivation (see module-doc block above).
# ---------------------------------------------------------------------------
CANONICAL_MODULES = {"core", "sql", "streaming", "pyspark", "docs"}

# First path segment -> canonical module.
_PATH_SEGMENT_MODULE = {
    "core": "core",
    "sql": "sql",
    "streaming": "streaming",
    "docs": "docs",
    "python": "pyspark",  # the python/pyspark source tree
}


def module_from_path(path: str | None) -> str | None:
    """Module from a file path / path_or_key via its first segment."""
    if not path:
        return None
    seg = path.split("/", 1)[0]
    return _PATH_SEGMENT_MODULE.get(seg)


def normalize_module(raw: str | None) -> str | None:
    """Normalize a legacy module token (repo_id suffix or jira `module` value).

    Handles 'spark:<m>', 'python', 'python/pyspark', already-canonical values,
    and returns NULL for anything unrecognized/unmapped.
    """
    if not raw:
        return None
    val = raw.strip().lower()
    if val.startswith("spark:"):
        val = val.split(":", 1)[1]
    if val in ("python", "python/pyspark"):
        return "pyspark"
    val = val.split("/", 1)[0]  # be liberal about any residual '<seg>/...'
    val = _PATH_SEGMENT_MODULE.get(val, val)
    return val if val in CANONICAL_MODULES else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def executemany_batched(cur, sql: str, rows: list, batch: int = BATCH) -> None:
    for i in range(0, len(rows), batch):
        cur.executemany(sql, rows[i : i + batch])


def jdumps(obj) -> str | None:
    return None if obj is None else json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-table loaders
# ---------------------------------------------------------------------------
def load_repos(cur) -> int:
    cur.execute(
        """
        INSERT INTO repos (org_id, repo_id, name, indexed_sha)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (repo_id) DO UPDATE
            SET org_id = EXCLUDED.org_id,
                name = EXCLUDED.name,
                indexed_sha = EXCLUDED.indexed_sha
        """,
        (ORG_ID, REPO_ID, REPO_NAME, INDEXED_SHA),
    )
    return 1


def load_documents(cur) -> int:
    rows = []
    for r in iter_jsonl(CATALOG_PATH):
        doc_id = r["doc_id"]
        path = r.get("path_or_key")
        kind = r.get("source_type")
        module = module_from_path(path)
        # Preserve provenance / legacy ids for later reconciliation work.
        meta = dict(r.get("metadata") or {})
        if r.get("text_path"):
            meta["text_path"] = r["text_path"]
        if r.get("repo_id"):
            meta["legacy_repo_id"] = r["repo_id"]
        rows.append(
            (ORG_ID, REPO_ID, doc_id, module, path, kind, jdumps(meta), r.get("content_hash"))
        )
    executemany_batched(
        cur,
        """
        INSERT INTO documents (org_id, repo_id, doc_id, module, path, kind, meta, content_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (repo_id, doc_id) DO UPDATE
            SET org_id = EXCLUDED.org_id,
                module = EXCLUDED.module,
                path = EXCLUDED.path,
                kind = EXCLUDED.kind,
                meta = EXCLUDED.meta,
                content_hash = EXCLUDED.content_hash
        """,
        rows,
    )
    return len(rows)


def load_nodes(cur) -> tuple[int, set]:
    rows = []
    node_ids: set[str] = set()
    for r in iter_jsonl(NODES_PATH):
        nid = r["id"]
        node_ids.add(nid)
        kind = r.get("type")
        path = r.get("file")
        module = module_from_path(path)
        meta = {k: v for k, v in r.items() if k not in ("id", "type", "file")}
        rows.append(
            (ORG_ID, REPO_ID, nid, nid, kind, module, path, jdumps(meta))
        )
    executemany_batched(
        cur,
        """
        INSERT INTO nodes (org_id, repo_id, id, local_uid, kind, module, path, meta)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (repo_id, id) DO UPDATE
            SET org_id = EXCLUDED.org_id,
                local_uid = EXCLUDED.local_uid,
                kind = EXCLUDED.kind,
                module = EXCLUDED.module,
                path = EXCLUDED.path,
                meta = EXCLUDED.meta
        """,
        rows,
    )
    return len(rows), node_ids


def load_edges(cur, node_ids: set) -> tuple[int, int]:
    """Reload edges (no natural key): delete this repo's edges then insert.

    is_resolved is true only when dst is an existing node id in the same repo.
    """
    rows = []
    resolved = 0
    for r in iter_jsonl(EDGES_PATH):
        src = r["src"]
        dst = r["dst"]
        kind = r.get("type")
        is_resolved = dst in node_ids
        if is_resolved:
            resolved += 1
        rows.append((ORG_ID, REPO_ID, src, dst, kind, is_resolved))
    cur.execute(
        "DELETE FROM edges WHERE org_id = %s AND repo_id = %s", (ORG_ID, REPO_ID)
    )
    executemany_batched(
        cur,
        """
        INSERT INTO edges (org_id, repo_id, src, dst, kind, is_resolved)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        rows,
    )
    return len(rows), resolved


def load_jira_module_links(cur) -> int:
    """Reload jira_module_links (no natural key): delete then insert."""
    rows = []
    for r in iter_jsonl(JIRA_MODULES_PATH):
        jira_key = r["jira_key"]
        module = normalize_module(r.get("module"))
        meta = {k: v for k, v in r.items() if k not in ("jira_key", "module")}
        # keep the raw module token for auditability when it was normalized
        if r.get("module") and r.get("module") != module:
            meta["module_raw"] = r.get("module")
        rows.append((ORG_ID, REPO_ID, jira_key, module, jdumps(meta)))
    cur.execute(
        "DELETE FROM jira_module_links WHERE org_id = %s AND repo_id = %s",
        (ORG_ID, REPO_ID),
    )
    executemany_batched(
        cur,
        """
        INSERT INTO jira_module_links (org_id, repo_id, jira_key, module, meta)
        VALUES (%s, %s, %s, %s, %s)
        """,
        rows,
    )
    return len(rows)


def load_chunks(cur) -> int:
    """Load chunks. embedding stays NULL; tsv is built with to_tsvector('english').

    The upsert intentionally does not touch embedding / embedding_version so a
    later backfill is preserved across loader re-runs.
    """
    rows = []
    for r in iter_jsonl(CHUNKS_PATH):
        chunk_id = r["chunk_id"]
        doc_id = r.get("doc_id")
        content = r.get("text")
        # chunks are Jira-only today; path_or_key is 'SPARK-nnnnn' (no module segment)
        module = module_from_path(r.get("path_or_key"))
        rows.append((ORG_ID, REPO_ID, chunk_id, doc_id, module, content, content))
    executemany_batched(
        cur,
        """
        INSERT INTO chunks (org_id, repo_id, chunk_id, doc_id, module, content, tsv)
        VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('english', %s))
        ON CONFLICT (repo_id, chunk_id) DO UPDATE
            SET org_id = EXCLUDED.org_id,
                doc_id = EXCLUDED.doc_id,
                module = EXCLUDED.module,
                content = EXCLUDED.content,
                tsv = EXCLUDED.tsv
        """,
        rows,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
TARGETS = ("repos", "documents", "nodes", "edges", "jira_module_links", "chunks")


def main() -> int:
    ap = argparse.ArgumentParser(description="Load JSONL data into the Graph-RAG Postgres store.")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=TARGETS,
        help="Load only these targets (default: all). Note: 'edges' requires 'nodes' to be present.",
    )
    ap.add_argument("--database-url", default=DATABASE_URL)
    args = ap.parse_args()

    selected = set(args.only) if args.only else set(TARGETS)

    print(f"Connecting to {args.database_url}")
    with psycopg.connect(args.database_url) as conn:
        # RLS is fail-closed on app.org_id. Set it (session scope) and commit so
        # it persists across the per-table transactions below.
        with conn.cursor() as cur:
            # set_config(..., is_local=false) => session scope, like SET.
            cur.execute("SELECT set_config('app.org_id', %s, false)", (ORG_ID,))
        conn.commit()

        counts: dict[str, int] = {}
        node_ids: set[str] = set()

        with conn.cursor() as cur:
            if "repos" in selected:
                counts["repos"] = load_repos(cur)
                conn.commit()
                print(f"  repos: {counts['repos']}")

            if "documents" in selected:
                counts["documents"] = load_documents(cur)
                conn.commit()
                print(f"  documents: {counts['documents']}")

            # nodes must be loaded (or read) before edges so is_resolved is correct.
            if "nodes" in selected or "edges" in selected:
                counts["nodes"], node_ids = load_nodes(cur)
                conn.commit()
                if "nodes" in selected:
                    print(f"  nodes: {counts['nodes']}")

            if "edges" in selected:
                n_edges, resolved = load_edges(cur, node_ids)
                counts["edges"] = n_edges
                conn.commit()
                print(f"  edges: {n_edges} (is_resolved=true: {resolved}, false: {n_edges - resolved})")

            if "jira_module_links" in selected:
                counts["jira_module_links"] = load_jira_module_links(cur)
                conn.commit()
                print(f"  jira_module_links: {counts['jira_module_links']}")

            if "chunks" in selected:
                counts["chunks"] = load_chunks(cur)
                conn.commit()
                print(f"  chunks: {counts['chunks']} (embedding left NULL — run backfill_embeddings.py)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
