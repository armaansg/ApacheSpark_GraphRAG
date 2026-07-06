#!/usr/bin/env python3
"""Standalone embedding backfill for chunks.embedding (Phase 2, DECOUPLED).

The Phase 2 loader (scripts/load_postgres.py) intentionally leaves
chunks.embedding NULL. This script fills it, and is kept separate on purpose so
that ingestion never depends on an embedding provider / quota.

It:
  1. selects chunks with a NULL embedding (optionally capped by --limit),
  2. computes embeddings with a PLUGGABLE embedder chosen by the EMBEDDER env
     var / --embedder flag,
  3. ASSERTS the embedder's output dimension == 1536 (the schema's vector size)
     and fails loudly otherwise,
  4. UPDATEs chunks.embedding + chunks.embedding_version.

Embedders (EMBEDDER=...):
  * openai   -> OpenAI-compatible HTTP API. Reads COMPANY_API_BASE,
               COMPANY_API_KEY, EMBEDDING_MODEL from the environment/.env (same
               convention as scripts/rag_common.py). POSTs to <BASE>/embeddings.
               NOTE: this calls a paid API — do not run in CI/tests.
  * sentence-transformers (aka "local", "st")
             -> local sentence-transformers model named by ST_MODEL
               (default: 'sentence-transformers/all-MiniLM-L6-v2'). Most ST
               models are NOT 1536-dim, so this will trip the dimension guard
               unless ST_MODEL is a 1536-dim model — that is the guard working.
  * hash     -> deterministic OFFLINE stub (no network, no deps). Produces
               1536-dim vectors from a hash of the text. For exercising the
               plumbing / smoke tests only — NOT semantically meaningful, never
               use for real retrieval.

Configuration:
    export DATABASE_URL=postgresql://spark:spark@localhost:5433/sparkctx
    export EMBEDDER=openai          # or sentence-transformers | hash
    # for openai:
    export COMPANY_API_BASE=... COMPANY_API_KEY=... EMBEDDING_MODEL=text-embedding-3-small
    # for sentence-transformers:
    export ST_MODEL=some/1536-dim-model

Usage:
    python scripts/backfill_embeddings.py --dry-run          # report how many need embedding
    python scripts/backfill_embeddings.py --embedder hash --limit 5   # smoke test
    python scripts/backfill_embeddings.py --embedder openai           # real backfill
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional for this standalone script
    pass

# The schema declares embedding vector(1536); anything else must fail loudly.
EXPECTED_DIM = 1536

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://spark:spark@localhost:5433/sparkctx"
)
ORG_ID = os.environ.get("SPARK_ORG_ID", "apache")
REPO_ID = os.environ.get("SPARK_REPO_ID", "apache/spark")

# Only embed NULL rows -> this query is what makes re-runs cheap and resumable.
NULL_SELECT_SQL = (
    "SELECT chunk_id, content FROM chunks "
    "WHERE embedding IS NULL AND repo_id = %s "
    "ORDER BY chunk_id LIMIT %s"
)
UPDATE_SQL = (
    "UPDATE chunks SET embedding = %s, embedding_version = %s "
    "WHERE repo_id = %s AND chunk_id = %s"
)


# ---------------------------------------------------------------------------
# Pluggable embedders
# ---------------------------------------------------------------------------
class OpenAIEmbedder:
    """OpenAI-compatible /embeddings endpoint (no SDK dependency; uses urllib)."""

    def __init__(self) -> None:
        self.base = os.environ["COMPANY_API_BASE"].rstrip("/")
        self.api_key = os.environ["COMPANY_API_KEY"]
        self.model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.version = f"openai:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # keep input order
        items = sorted(data["data"], key=lambda d: d["index"])
        return [it["embedding"] for it in items]


class SentenceTransformersEmbedder:
    """Local sentence-transformers model (lazy import so it's optional)."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = os.environ.get(
            "ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)
        self.version = f"sentence-transformers:{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class HashEmbedder:
    """Deterministic OFFLINE stub — NOT semantic. For smoke tests / plumbing only."""

    def __init__(self, dim: int = EXPECTED_DIM) -> None:
        self.dim = dim
        self.version = f"hash-stub:dim{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            floats = []
            counter = 0
            while len(floats) < self.dim:
                h = hashlib.sha256(f"{counter}:{t}".encode("utf-8")).digest()
                # 8 float32s per 32-byte digest
                for i in range(0, 32, 4):
                    (val,) = struct.unpack("f", h[i : i + 4])
                    # squash NaN/inf into a finite range
                    if val != val or val in (float("inf"), float("-inf")):
                        val = 0.0
                    floats.append(float(val))
                counter += 1
            out.append(floats[: self.dim])
        return out


_EMBEDDERS = {
    "openai": OpenAIEmbedder,
    "sentence-transformers": SentenceTransformersEmbedder,
    "st": SentenceTransformersEmbedder,
    "local": SentenceTransformersEmbedder,
    "hash": HashEmbedder,
}


def get_embedder(name: str | None = None):
    name = (name or os.environ.get("EMBEDDER", "openai")).lower()
    if name not in _EMBEDDERS:
        raise SystemExit(
            f"Unknown embedder {name!r}. Choose one of: {sorted(_EMBEDDERS)}"
        )
    return _EMBEDDERS[name]()


# ---------------------------------------------------------------------------
# Dimension guard
# ---------------------------------------------------------------------------
def assert_dim(vectors: list[list[float]], expected: int = EXPECTED_DIM,
               embedder_name: str = "embedder") -> None:
    """Fail loudly unless every vector has exactly `expected` dimensions."""
    if not vectors:
        return
    dims = {len(v) for v in vectors}
    if dims != {expected}:
        raise ValueError(
            f"Embedding dimension mismatch: {embedder_name} produced vectors of "
            f"dimension {sorted(dims)}, but the schema column is vector({expected}). "
            f"Refusing to write. Pick a {expected}-dim model (e.g. OpenAI "
            f"text-embedding-3-small) or fix ST_MODEL."
        )


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------
def count_null(cur) -> int:
    return cur.execute(
        "SELECT count(*) FROM chunks WHERE embedding IS NULL AND repo_id = %s",
        (REPO_ID,),
    ).fetchone()[0]


def backfill(conn, embedder, batch_size: int, limit: int | None) -> int:
    total = 0
    embedder_name = type(embedder).__name__
    with conn.cursor() as cur:
        remaining = limit
        while True:
            take = batch_size if remaining is None else min(batch_size, remaining)
            if take <= 0:
                break
            rows = cur.execute(NULL_SELECT_SQL, (REPO_ID, take)).fetchall()
            if not rows:
                break
            ids = [r[0] for r in rows]
            texts = [r[1] or "" for r in rows]
            vectors = embedder.embed(texts)
            assert_dim(vectors, EXPECTED_DIM, embedder_name)
            with conn.cursor() as wcur:
                wcur.executemany(
                    UPDATE_SQL,
                    [
                        (vec, embedder.version, REPO_ID, cid)
                        for cid, vec in zip(ids, vectors)
                    ],
                )
            conn.commit()
            total += len(rows)
            if remaining is not None:
                remaining -= len(rows)
            print(f"  embedded {total} chunks (version={embedder.version})")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill NULL chunk embeddings.")
    ap.add_argument("--embedder", default=None,
                    help="openai | sentence-transformers | hash (default: $EMBEDDER or openai)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max chunks to embed this run (default: all NULL).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report how many chunks still need an embedding.")
    ap.add_argument("--database-url", default=DATABASE_URL)
    args = ap.parse_args()

    with psycopg.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            # RLS is fail-closed on app.org_id — set it before touching chunks.
            cur.execute("SELECT set_config('app.org_id', %s, false)", (ORG_ID,))
        conn.commit()
        register_vector(conn)

        with conn.cursor() as cur:
            pending = count_null(cur)
        print(f"chunks needing embedding: {pending}")
        if args.dry_run:
            return 0
        if pending == 0:
            print("Nothing to do.")
            return 0

        embedder = get_embedder(args.embedder)
        print(f"Using embedder: {type(embedder).__name__} (version={embedder.version})")
        done = backfill(conn, embedder, args.batch_size, args.limit)
        print(f"Done. Embedded {done} chunks. Remaining NULL: {count_null(conn.cursor())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
