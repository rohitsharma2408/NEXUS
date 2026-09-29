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

CREATE INDEX idx_business_documents_embedding
    ON business_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

GRANT SELECT ON business_documents TO nexus_readonly;
