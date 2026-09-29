"""RAG Agent — retrieves business documents (policies/reports) from pgvector, with citations."""
import os
from dataclasses import dataclass

from sqlalchemy import create_engine, text

from config import READONLY_DATABASE_URL

EMBEDDING_DIM = 768  # matches gemini-embedding-001 output_dimensionality=768, see pgvector_setup.sql


def _embed(text_query: str) -> list[float]:
    """Swap for your embedding provider. Kept provider-agnostic on purpose."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    if provider == "gemini" or os.environ.get("GOOGLE_API_KEY"):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        resp = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text_query,
            output_dimensionality=EMBEDDING_DIM,
        )
        return resp["embedding"]
    if provider == "openai" or os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(model="text-embedding-3-small", input=text_query)
        return resp.data[0].embedding
    raise RuntimeError(
        "No embedding provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY, or swap "
        "_embed() for your preferred embedding model."
    )


@dataclass
class RAGResult:
    chunks: list  # [{"document": str, "content": str, "score": float}]


def retrieve(question: str, k: int = 5) -> RAGResult:
    vector = _embed(question)
    engine = create_engine(READONLY_DATABASE_URL)
    sql = text("""
        SELECT document_name, content, 1 - (embedding <=> :vector) AS score
        FROM business_documents
        ORDER BY embedding <=> :vector
        LIMIT :k
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"vector": str(vector), "k": k}).mappings().all()
    chunks = [{"document": r["document_name"], "content": r["content"], "score": float(r["score"])} for r in rows]
    return RAGResult(chunks=chunks)
