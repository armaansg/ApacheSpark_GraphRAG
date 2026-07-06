#!/usr/bin/env python3
"""Tests for the standalone embedding backfill (Phase 2).

Covers the two things called out in the Phase 2 spec:
  * the dimension check (assert_dim) — must reject != 1536 and accept 1536,
  * the NULL-selection query — must return only chunks whose embedding is NULL.

Plus a fully offline end-to-end smoke test of the backfill loop using the
deterministic HashEmbedder, so the UPDATE path is exercised WITHOUT any API.

The DB-touching tests skip automatically if the live DB is unreachable, so this
file is safe to run in a bare CI environment. No paid API is ever called.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_embeddings import (  # noqa: E402
    EXPECTED_DIM,
    NULL_SELECT_SQL,
    HashEmbedder,
    assert_dim,
    get_embedder,
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://spark:spark@localhost:5433/sparkctx"
)
ORG_ID = os.environ.get("SPARK_ORG_ID", "apache")
REPO_ID = os.environ.get("SPARK_REPO_ID", "apache/spark")


def _connect():
    """Return an RLS-configured connection, or None if the DB is unreachable."""
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except Exception:
        return None
    try:
        conn = psycopg.connect(DATABASE_URL, connect_timeout=3)
    except Exception:
        return None
    conn.execute("SELECT set_config('app.org_id', %s, false)", (ORG_ID,))
    register_vector(conn)
    return conn


class TestDimensionCheck(unittest.TestCase):
    def test_accepts_expected_dim(self):
        # exactly 1536 -> no raise
        assert_dim([[0.0] * EXPECTED_DIM], EXPECTED_DIM, "test")

    def test_rejects_wrong_dim(self):
        with self.assertRaises(ValueError) as ctx:
            assert_dim([[0.0] * 384], EXPECTED_DIM, "MiniLM")
        msg = str(ctx.exception)
        self.assertIn("384", msg)
        self.assertIn("1536", msg)

    def test_rejects_mixed_dims(self):
        with self.assertRaises(ValueError):
            assert_dim([[0.0] * 1536, [0.0] * 1535], EXPECTED_DIM, "test")

    def test_empty_is_noop(self):
        assert_dim([], EXPECTED_DIM, "test")  # must not raise

    def test_hash_embedder_is_1536_dim(self):
        vecs = get_embedder("hash").embed(["hello", "world"])
        self.assertEqual({len(v) for v in vecs}, {EXPECTED_DIM})
        assert_dim(vecs, EXPECTED_DIM, "hash")

    def test_hash_embedder_deterministic(self):
        a = HashEmbedder().embed(["same text"])[0]
        b = HashEmbedder().embed(["same text"])[0]
        self.assertEqual(a, b)


class TestNullSelectionQuery(unittest.TestCase):
    def setUp(self):
        self.conn = _connect()
        if self.conn is None:
            self.skipTest("live DB not reachable")

    def tearDown(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()

    def test_selects_only_null_embeddings(self):
        rows = self.conn.execute(NULL_SELECT_SQL, (REPO_ID, 50)).fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            self.skipTest("no NULL-embedding chunks to check")
        # every returned chunk must actually have a NULL embedding
        placeholders = ",".join(["%s"] * len(ids))
        not_null = self.conn.execute(
            f"SELECT count(*) FROM chunks WHERE repo_id = %s AND chunk_id IN ({placeholders}) "
            f"AND embedding IS NOT NULL",
            (REPO_ID, *ids),
        ).fetchone()[0]
        self.assertEqual(not_null, 0)

    def test_count_matches_null_count(self):
        # the LIMIT-less count of NULL rows should equal a broad selection size
        total_null = self.conn.execute(
            "SELECT count(*) FROM chunks WHERE embedding IS NULL AND repo_id = %s",
            (REPO_ID,),
        ).fetchone()[0]
        selected = self.conn.execute(NULL_SELECT_SQL, (REPO_ID, 10_000_000)).fetchall()
        self.assertEqual(len(selected), total_null)


class TestBackfillSmokeOffline(unittest.TestCase):
    """End-to-end backfill on a couple of rows using the offline HashEmbedder."""

    def setUp(self):
        self.conn = _connect()
        if self.conn is None:
            self.skipTest("live DB not reachable")
        self.touched = []

    def tearDown(self):
        # always reset any rows we embedded back to NULL to keep the DB pristine
        if getattr(self, "conn", None) is not None:
            if self.touched:
                ph = ",".join(["%s"] * len(self.touched))
                self.conn.execute(
                    f"UPDATE chunks SET embedding = NULL, embedding_version = NULL "
                    f"WHERE repo_id = %s AND chunk_id IN ({ph})",
                    (REPO_ID, *self.touched),
                )
                self.conn.commit()
            self.conn.close()

    def test_backfill_then_reset(self):
        from backfill_embeddings import backfill

        pending = self.conn.execute(
            "SELECT count(*) FROM chunks WHERE embedding IS NULL AND repo_id = %s",
            (REPO_ID,),
        ).fetchone()[0]
        if pending == 0:
            self.skipTest("no NULL chunks to backfill")

        # remember which ids we will touch so tearDown can null them out
        self.touched = [
            r[0]
            for r in self.conn.execute(NULL_SELECT_SQL, (REPO_ID, 3)).fetchall()
        ]
        done = backfill(self.conn, HashEmbedder(), batch_size=3, limit=3)
        self.assertEqual(done, len(self.touched))
        # confirm they now have a non-null embedding + version stamp
        ph = ",".join(["%s"] * len(self.touched))
        n_set = self.conn.execute(
            f"SELECT count(*) FROM chunks WHERE repo_id = %s AND chunk_id IN ({ph}) "
            f"AND embedding IS NOT NULL AND embedding_version LIKE 'hash-stub%%'",
            (REPO_ID, *self.touched),
        ).fetchone()[0]
        self.assertEqual(n_set, len(self.touched))


if __name__ == "__main__":
    unittest.main(verbosity=2)
