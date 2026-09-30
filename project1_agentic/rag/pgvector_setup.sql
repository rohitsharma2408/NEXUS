-- Run once against nexus_db (superuser) to enable RAG.
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS business_documents;

CREATE TABLE business_documents (
    id BIGSERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,  -- matches gemini-embedding-001 output_dimensionality=768
    created_at TIMESTAMPTZ DEFAULT now()
);

-- No ANN index on purpose: an IVFFlat index with lists=100 on a tiny table returns EMPTY results
-- (most clusters are empty). Exact scan is instant at this scale. Add an HNSW index only once
-- the table holds thousands of chunks:
--   CREATE INDEX ON business_documents USING hnsw (embedding vector_cosine_ops);

GRANT SELECT ON business_documents TO nexus_readonly;
