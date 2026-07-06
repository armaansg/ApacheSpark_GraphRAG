# Phase 2 — JSONL → Postgres data loader

ETL layer that ingests the normalized JSONL files under `data/normalized/` into
the Phase 1 Postgres 16 + pgvector schema (`schema.sql`). Query-side code is
untouched here — that is Phase 3.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/load_postgres.py` | Loads all six targets end-to-end (idempotent). |
| `scripts/backfill_embeddings.py` | Standalone, pluggable embedding backfill for `chunks.embedding` (decoupled from the loader). |
| `scripts/test_backfill_embeddings.py` | Unit tests for the dimension check + NULL-selection query, plus an offline end-to-end smoke test. |

## Prerequisites

```bash
pip install -r requirements.txt          # psycopg[binary], pgvector, python-dotenv
# schema.sql must already be applied to the target database (Phase 1).
export DATABASE_URL=postgresql://spark:spark@localhost:5433/sparkctx   # default if unset
```

## Connection & RLS

Every table has **RLS ENABLED and FORCED** with a **fail-closed** policy comparing
`org_id` to the per-connection GUC `app.org_id`. If the GUC is unset,
`current_setting('app.org_id', true)` is `NULL`, so `org_id = NULL` is never true
and **all reads and writes are denied**.

Both scripts therefore run, immediately after connecting:

```sql
SELECT set_config('app.org_id', 'apache', false);   -- session scope, like SET
```

and stamp `org_id = 'apache'` on every row. Without both, inserts are silently
rejected by the policy's `WITH CHECK`.

## Locked data-model decisions

- `apache/spark` is **one repo**: `repo_id = 'apache/spark'` on every row.
- `org_id = 'apache'` (constant) on every row.
- `module ∈ {core, sql, streaming, pyspark, docs}` is a **column**, derived per row.

### path → module mapping (the reconciliation rules)

First path segment (used for `documents` from `path_or_key`, `nodes` from `file`):

| segment | module |
| --- | --- |
| `core` | core |
| `sql` | sql |
| `streaming` | streaming |
| `docs` | docs |
| `python` | **pyspark**  (the `python/pyspark` source tree) |

Legacy `repo_id` on catalog code_file rows (`spark:<module>`):

| repo_id | module |
| --- | --- |
| `spark:core` | core |
| `spark:sql` | sql |
| `spark:streaming` | streaming |
| `spark:docs` | docs |
| `spark:pyspark` | pyspark |

Jira `module` values in `jira_modules.jsonl`:

| raw value | module |
| --- | --- |
| `python/pyspark` | **pyspark** (reconciled to match a `python/...` path) |
| `core` / `sql` / `streaming` / `docs` / `pyspark` | itself |
| `null` / unmapped component | `NULL` |

`jira_issue` documents have `path_or_key = 'SPARK-nnnnn'` (no module segment), so
their `documents.module` is `NULL`; the authoritative Jira→module mapping lives in
`jira_module_links` (a single Jira issue can map to several modules).

## Source → target mapping

| source (`data/normalized/…`) | target table | notes |
| --- | --- | --- |
| `catalog.jsonl` | `documents` | `source_type`→`kind`, `metadata`(+`text_path`,`legacy_repo_id`)→`meta` |
| `graph/nodes.jsonl` | `nodes` | `id`→`id`&`local_uid`, `type`→`kind`, `file`→`path`, rest→`meta` |
| `graph/edges.jsonl` | `edges` | `type`→`kind`; `is_resolved=true` iff `dst` is an existing node id |
| `links/jira_modules.jsonl` | `jira_module_links` | `module` normalized; rest→`meta` |
| `chunks/chunks.jsonl` | `chunks` | `text`→`content`, `tsv=to_tsvector('english',content)`, **`embedding` left NULL** |
| (synthesized) | `repos` | single `apache/spark` row, `indexed_sha = v3.5.4-rc3` (from catalog `metadata.ref`) |

## Idempotency

- Natural-key tables (`repos`, `documents`, `nodes`, `chunks`) use
  `INSERT … ON CONFLICT (<pk>) DO UPDATE`. The `chunks` upsert deliberately does
  **not** overwrite `embedding` / `embedding_version`, so a later backfill
  survives a loader re-run.
- No-key tables (`edges`, `jira_module_links` — duplicate rows are legitimate)
  are reloaded with `DELETE (this org+repo)` then `INSERT` in one transaction.

Running the loader twice leaves row counts unchanged.

## Run order & usage

```bash
python scripts/load_postgres.py                     # load everything
python scripts/load_postgres.py --only nodes edges  # subset (edges pulls node ids for is_resolved)
```

Load order: `repos → documents → nodes → edges → jira_module_links → chunks`
(nodes before edges so `is_resolved` is computed against the node-id set).

## Embeddings — decoupled backfill

The loader never calls an embedding API; `chunks.embedding` stays `NULL`. Fill it
separately:

```bash
python scripts/backfill_embeddings.py --dry-run                 # how many need embedding
python scripts/backfill_embeddings.py --embedder openai         # real backfill (paid API)
python scripts/backfill_embeddings.py --embedder hash --limit 5 # offline smoke test
```

Pluggable embedder chosen by `--embedder` / `$EMBEDDER`:

| embedder | source | config |
| --- | --- | --- |
| `openai` | OpenAI-compatible `/embeddings` HTTP API | `COMPANY_API_BASE`, `COMPANY_API_KEY`, `EMBEDDING_MODEL` |
| `sentence-transformers` (`st`, `local`) | local sentence-transformers model | `ST_MODEL` (must be 1536-dim) |
| `hash` | deterministic **offline stub**, not semantic — plumbing/tests only | — |

The script **asserts the embedder's output dimension == 1536** (the schema's
`vector(1536)`) and fails loudly otherwise. Only `NULL`-embedding rows are
selected, so the backfill is resumable and cheap to re-run.

Run the tests with:

```bash
python scripts/test_backfill_embeddings.py
```
