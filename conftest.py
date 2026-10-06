"""Repo-wide pytest setup: import paths, and an env default so offline tests can import config."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "project1_agentic", ROOT / "project1_agentic" / "agents",
          ROOT / "project2_analytics" / "ml", ROOT / "evaluation"):
    sys.path.insert(0, str(p))

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://nexus_user:change_me@localhost:5432/nexus_db")
os.environ.setdefault("READONLY_DATABASE_URL",
                      "postgresql+psycopg2://nexus_readonly:change_me_too@localhost:5432/nexus_db")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-used")
# .env points MODEL_DIR at the container path (/app/models); tests use the repo-local directory
os.environ["MODEL_DIR"] = os.environ.get("TEST_MODEL_DIR", str(ROOT / "models"))
