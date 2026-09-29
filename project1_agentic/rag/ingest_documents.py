"""
Loads business documents (policies, reports, promo calendars, etc.) from
project1_agentic/rag/documents/*.txt, chunks them, embeds them, and stores them in
pgvector's business_documents table for the RAG Agent.

Usage:
    python ingest_documents.py --docs-dir project1_agentic/rag/documents
"""
import argparse
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_DIM = 768  # must match project1_agentic/agents/rag_agent.py and pgvector_setup.sql


def chunk_text(text_str: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text_str):
        end = start + size
        chunks.append(text_str[start:end])
        start = end - overlap
    return chunks


def embed(text_str: str) -> list[float]:
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    if provider == "gemini" or os.environ.get("GOOGLE_API_KEY"):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        resp = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text_str,
            output_dimensionality=EMBEDDING_DIM,
        )
        return resp["embedding"]
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model="text-embedding-3-small", input=text_str)
    return resp.data[0].embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default=str(Path(__file__).parent / "documents"))
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(os.environ["DATABASE_URL"])
    files = list(docs_dir.glob("*.txt"))
    if not files:
        print(f"No .txt documents found in {docs_dir}. Add policy/report text files and re-run.")
        return

    with engine.begin() as conn:
        for f in files:
            content = f.read_text()
            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                vector = embed(chunk)
                conn.execute(
                    text("""
                        INSERT INTO business_documents (document_name, chunk_index, content, embedding)
                        VALUES (:name, :idx, :content, :embedding)
                    """),
                    {"name": f.name, "idx": i, "content": chunk, "embedding": str(vector)},
                )
            print(f"Ingested {f.name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
