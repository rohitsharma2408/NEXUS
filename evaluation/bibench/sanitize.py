"""Matches tools/run_large_models.py's sanitize_name so table/column names are consistent
with how the paper's own harness builds each case's per-question SQLite database."""
import re


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if re.match(r"^\d", name):
        name = "_" + name
    return name or "_col"
