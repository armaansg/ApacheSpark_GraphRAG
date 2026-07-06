-- =============================================================================
-- Spark-Context Graph-RAG store — Phase 1 schema
-- Target: PostgreSQL 16 + pgvector (image: pgvector/pgvector:pg16)
--
-- This file is DDL only. It is written to apply cleanly top-to-bottom and is
-- idempotent where practical (CREATE ... IF NOT EXISTS, DROP POLICY IF EXISTS).
--
-- Multi-tenancy model (locked decisions)
-- --------------------------------------
--   * EVERY table carries `org_id text NOT NULL` and (except memories) `repo_id
--     text NOT NULL`. org_id is a hard tenant boundary enforced by RLS.
--   * `apache/spark` is ONE repo (repo_id = 'apache/spark'). Its sub-projects
--     (core, sql, streaming, pyspark, docs) are a first-class `module` COLUMN,
--     never separate repos.
--   * org_id is the constant 'apache' for now, but the column exists everywhere
--     so a second org can be added without a migration.
--   * repo_id is the routing/versioning anchor. The `repos` table records the
--     indexed_sha so retrieval can be pinned to a known commit; this is what the
--     later hardcoded-path removal keys off of.
--
-- Partitioning
-- ------------
--   The four large tables (documents, nodes, edges, chunks) are partitioned
--   BY HASH(repo_id) into 4 partitions as an illustrative default. See the
--   NOTE block above `documents` for the PK-vs-partition-key tradeoff.
-- =============================================================================

-- pgvector — provides the `vector` type and the HNSW / IVF index access methods.
CREATE EXTENSION IF NOT EXISTS vector;

-- gen_random_uuid() is in PostgreSQL core since v13; no extension required.


-- -----------------------------------------------------------------------------
-- repos — tenant/repo registry. The routing & versioning anchor.
-- Not partitioned: tiny (one row per indexed repo).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS repos (
    org_id       text        NOT NULL,
    repo_id      text        NOT NULL,
    name         text        NOT NULL,
    indexed_sha  text,                                   -- commit the index was built from
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (repo_id)
);


-- -----------------------------------------------------------------------------
-- documents — retrieval document registry (source: catalog.jsonl, ~5,592 rows).
--
-- NOTE (partitioning-vs-PK tradeoff):
--   PostgreSQL requires every column of a partitioned table's PRIMARY KEY /
--   UNIQUE constraint to be a superset of the partition key. We partition BY
--   HASH(repo_id), and the natural business key here is doc_id. We therefore
--   make the PK composite: (repo_id, doc_id). Consequences:
--     * (repo_id, doc_id) is unique — correct, since ids are only ever resolved
--       within a repo.
--     * A bare doc_id is NOT globally unique across repos. Acceptable and in
--       fact desirable: two repos may legitimately share a doc_id string.
--   No compromise on intra-repo uniqueness was needed because repo_id is a
--   member of the composite PK — HASH(repo_id) partitioning is fully compatible.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    org_id        text  NOT NULL,
    repo_id       text  NOT NULL,
    doc_id        text  NOT NULL,
    module        text,                                  -- core | sql | streaming | pyspark | docs
    path          text,
    kind          text,                                  -- e.g. code_file, jira_issue
    meta          jsonb,
    content_hash  text,
    PRIMARY KEY (repo_id, doc_id)
) PARTITION BY HASH (repo_id);

CREATE TABLE IF NOT EXISTS documents_p0 PARTITION OF documents FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE IF NOT EXISTS documents_p1 PARTITION OF documents FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE IF NOT EXISTS documents_p2 PARTITION OF documents FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE IF NOT EXISTS documents_p3 PARTITION OF documents FOR VALUES WITH (MODULUS 4, REMAINDER 3);


-- -----------------------------------------------------------------------------
-- nodes — code graph nodes (source: nodes.jsonl, ~59,763 rows).
-- Composite PK (repo_id, id): `id` (e.g. "file:...", "class:...") is unique
-- within a repo; repo_id is included to satisfy the HASH(repo_id) partition key.
-- local_uid holds the repo-local identifier as ingested, kept distinct from the
-- (future) globally-qualified id so cross-repo work does not need a rewrite.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nodes (
    org_id     text  NOT NULL,
    repo_id    text  NOT NULL,
    id         text  NOT NULL,
    local_uid  text,
    kind       text,                                     -- file | class | method | symbol | ...
    module     text,
    path       text,
    meta       jsonb,
    PRIMARY KEY (repo_id, id)
) PARTITION BY HASH (repo_id);

CREATE TABLE IF NOT EXISTS nodes_p0 PARTITION OF nodes FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE IF NOT EXISTS nodes_p1 PARTITION OF nodes FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE IF NOT EXISTS nodes_p2 PARTITION OF nodes FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE IF NOT EXISTS nodes_p3 PARTITION OF nodes FOR VALUES WITH (MODULUS 4, REMAINDER 3);


-- -----------------------------------------------------------------------------
-- edges — code graph edges (source: edges.jsonl, ~111,297 rows).
--
-- IMPORTANT: `dst` is PLAIN TEXT with NO foreign key. ~39k dst values are
-- synthetic (e.g. "symbol:...", "call:...") and intentionally do NOT exist in
-- `nodes`. Adding an FK would reject those rows. `src` is likewise left
-- FK-free for symmetry and load robustness. Both endpoints are same-repo, so
-- repo_id qualifies both.
--
-- No natural PK (edges may legitimately repeat with different kinds); we do not
-- declare one. Supporting btree indexes below cover both traversal directions.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS edges (
    org_id       text    NOT NULL,
    repo_id      text    NOT NULL,
    src          text    NOT NULL,
    dst          text    NOT NULL,
    kind         text,                                   -- CONTAINS | CALLS | REFERENCES | ...
    is_resolved  boolean NOT NULL DEFAULT false          -- true once dst is confirmed to be a real node
) PARTITION BY HASH (repo_id);

CREATE TABLE IF NOT EXISTS edges_p0 PARTITION OF edges FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE IF NOT EXISTS edges_p1 PARTITION OF edges FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE IF NOT EXISTS edges_p2 PARTITION OF edges FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE IF NOT EXISTS edges_p3 PARTITION OF edges FOR VALUES WITH (MODULUS 4, REMAINDER 3);


-- -----------------------------------------------------------------------------
-- chunks — retrievable text units (source: chunks.jsonl, ~2,388; Jira-only today).
-- Composite PK (repo_id, chunk_id) for the same partition-key reason as above.
-- `doc_id` is a soft reference to documents(doc_id) — deliberately NOT an FK,
-- because today's chunks are Jira issues whose doc_id may not be present in the
-- catalog-derived `documents` table. embedding is a 1536-dim vector (OpenAI
-- text-embedding-3-small default); embedding_version records the model/version
-- so re-embeddings can be tracked without dropping rows.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    org_id             text          NOT NULL,
    repo_id            text          NOT NULL,
    chunk_id           text          NOT NULL,
    doc_id             text,                             -- soft ref to documents.doc_id (no FK; see note)
    module             text,
    content            text,
    tsv                tsvector,
    embedding          vector(1536),
    embedding_version  text,
    content_hash       text,
    PRIMARY KEY (repo_id, chunk_id)
) PARTITION BY HASH (repo_id);

CREATE TABLE IF NOT EXISTS chunks_p0 PARTITION OF chunks FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE IF NOT EXISTS chunks_p1 PARTITION OF chunks FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE IF NOT EXISTS chunks_p2 PARTITION OF chunks FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE IF NOT EXISTS chunks_p3 PARTITION OF chunks FOR VALUES WITH (MODULUS 4, REMAINDER 3);


-- -----------------------------------------------------------------------------
-- jira_module_links — intra-repo Jira-issue -> module edges
-- (source: jira_modules.jsonl, ~2,932 rows). Not partitioned: small.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jira_module_links (
    org_id     text  NOT NULL,
    repo_id    text  NOT NULL,
    jira_key   text  NOT NULL,                           -- e.g. SPARK-17091
    module     text,                                     -- core | sql | streaming | pyspark | docs
    meta       jsonb
);


-- -----------------------------------------------------------------------------
-- memories — 3-tier agentic memory (session / repo / org).
-- repo_id is NULLABLE so an org-level memory can exist without any repo; org_id
-- is always required. Not partitioned. `tier` names the scope level and `scope`
-- holds the concrete scope value (e.g. a session id) it is bound to.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id          bigint       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id      text         NOT NULL,
    repo_id     text,                                    -- NULL => org-level memory
    tier        text         NOT NULL,                   -- 'session' | 'repo' | 'org'
    scope       text,                                    -- concrete binding for the tier (e.g. session id)
    content     text,
    embedding   vector(1536),
    created_at  timestamptz  NOT NULL DEFAULT now()
);


-- =============================================================================
-- FUTURE / STUB TABLES — intentionally NOT created in Phase 1.
-- These stay commented out until a SECOND repo exists; with a single repo they
-- would only ever be empty. Definitions are recorded here so the shape is
-- agreed now and enabling them later is a one-line uncomment + partition setup.
-- =============================================================================
--
-- cross_repo_edges — edges whose endpoints live in DIFFERENT repos (e.g. a
-- shared-library symbol used across repos). Distinct from `edges`, which is
-- strictly same-repo. Would carry org_id on both sides for tenant isolation.
--
-- CREATE TABLE cross_repo_edges (
--     org_id       text    NOT NULL,
--     src_repo_id  text    NOT NULL,
--     src          text    NOT NULL,
--     dst_repo_id  text    NOT NULL,
--     dst          text    NOT NULL,
--     kind         text,
--     is_resolved  boolean NOT NULL DEFAULT false
-- ) PARTITION BY HASH (org_id);   -- partition on org_id: edges span repos, not one
--
-- repo_summaries — per-repo rollups (module inventory, embedding coverage,
-- last-indexed stats) used for cross-repo routing/ranking once N > 1.
--
-- CREATE TABLE repo_summaries (
--     org_id      text        NOT NULL,
--     repo_id     text        NOT NULL,
--     summary     text,
--     stats       jsonb,
--     embedding   vector(1536),
--     updated_at  timestamptz NOT NULL DEFAULT now(),
--     PRIMARY KEY (repo_id)
-- );


-- =============================================================================
-- INDEXES
-- Indexes on a partitioned parent are created as partitioned indexes and cascade
-- to every partition (existing and future). CREATE INDEX IF NOT EXISTS keeps
-- this section idempotent.
-- =============================================================================

-- Required: full-text search over chunk content.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Required: approximate nearest-neighbour over chunk embeddings (cosine).
-- HNSW on a partitioned table builds one HNSW index per partition.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Supporting indexes for common access paths (pure DDL; the store is unusable
-- for graph traversal / lookup without them).
CREATE INDEX IF NOT EXISTS edges_src_idx      ON edges (repo_id, src);   -- outgoing traversal
CREATE INDEX IF NOT EXISTS edges_dst_idx      ON edges (repo_id, dst);   -- incoming traversal
CREATE INDEX IF NOT EXISTS nodes_module_idx   ON nodes (repo_id, module);
CREATE INDEX IF NOT EXISTS documents_module_idx ON documents (repo_id, module);
CREATE INDEX IF NOT EXISTS chunks_doc_idx     ON chunks (repo_id, doc_id);
CREATE INDEX IF NOT EXISTS jira_module_links_key_idx ON jira_module_links (repo_id, jira_key);
CREATE INDEX IF NOT EXISTS memories_scope_idx ON memories (org_id, repo_id, tier);


-- =============================================================================
-- ROW-LEVEL SECURITY
--
-- Tenant isolation is enforced on org_id. Every table gets RLS ENABLED and
-- FORCED (FORCE so that even the table owner is subject to the policy — without
-- it, the owner role silently bypasses RLS).
--
-- The policy compares org_id to the per-connection GUC `app.org_id`. The
-- application is expected to set this once per connection / transaction, e.g.:
--
--     SET app.org_id = 'apache';                 -- session scope
--     -- or, preferred for pooled connections:
--     SELECT set_config('app.org_id', 'apache', true);   -- transaction scope
--
-- current_setting('app.org_id', true) uses missing_ok = true: if the GUC was
-- never set it returns NULL, and `org_id = NULL` is never true, so the default
-- posture is deny-all (fail closed) rather than leak-all.
--
-- On partitioned tables, RLS enabled on the parent is enforced for all access
-- through the parent; policies do not need to be repeated per partition.
-- =============================================================================

-- repos
ALTER TABLE repos ENABLE ROW LEVEL SECURITY;
ALTER TABLE repos FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS repos_org_isolation ON repos;
CREATE POLICY repos_org_isolation ON repos
    USING (org_id = current_setting('app.org_id', true))
    WITH CHECK (org_id = current_setting('app.org_id', true));

-- documents
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS documents_org_isolation ON documents;
CREATE POLICY documents_org_isolation ON documents
    USING (org_id = current_setting('app.org_id', true))
    WITH CHECK (org_id = current_setting('app.org_id', true));

-- nodes
ALTER TABLE nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE nodes FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS nodes_org_isolation ON nodes;
CREATE POLICY nodes_org_isolation ON nodes
    USING (org_id = current_setting('app.org_id', true))
    WITH CHECK (org_id = current_setting('app.org_id', true));

-- edges
ALTER TABLE edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE edges FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS edges_org_isolation ON edges;
CREATE POLICY edges_org_isolation ON edges
    USING (org_id = current_setting('app.org_id', true))
    WITH CHECK (org_id = current_setting('app.org_id', true));

-- chunks
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS chunks_org_isolation ON chunks;
CREATE POLICY chunks_org_isolation ON chunks
    USING (org_id = current_setting('app.org_id', true))
    WITH CHECK (org_id = current_setting('app.org_id', true));

-- jira_module_links
ALTER TABLE jira_module_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE jira_module_links FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS jira_module_links_org_isolation ON jira_module_links;
CREATE POLICY jira_module_links_org_isolation ON jira_module_links
    USING (org_id = current_setting('app.org_id', true))
    WITH CHECK (org_id = current_setting('app.org_id', true));

-- memories
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS memories_org_isolation ON memories;
CREATE POLICY memories_org_isolation ON memories
    USING (org_id = current_setting('app.org_id', true))
    WITH CHECK (org_id = current_setting('app.org_id', true));

-- =============================================================================
-- End of Phase 1 schema.
-- =============================================================================
