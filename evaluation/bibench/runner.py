"""
BI-Bench external-schema pilot runner.

Unlike project1_agentic/agents/sql_agent.py (which has ALLOWED_TABLES and a SCHEMA_SUMMARY
hardcoded for the one Global E-Commerce warehouse), a BI-Bench case is a different company
every time, with its own tables the agent has never seen. This script is what the blueprint's
Section 5 external validation actually requires: it loads a case's raw CSVs into a fresh
per-case SQLite database (matching the paper's own harness — sanitize_name, dtype=str), reads
the schema BACK from the database instead of assuming it, asks the same LLM configured in
project1_agentic/config.py to write SQL against that discovered schema, executes it, and scores
the result against the case's ground truth using the paper's own comparison logic (scoring.py).

Usage:
    python evaluation/bibench/runner.py --data-dir evaluation/bibench/pilot_data --limit 20
    python evaluation/bibench/runner.py --data-dir evaluation/bibench/pilot_data --case 106024162

This is a PILOT on a small, size-capped subset (see docs/bibench_pilot.md for how the 20 cases
were sampled) — not the full 100-case benchmark, and the paper's own "with tools" mode (join
graph, transform library) is not reproduced here. Report it as a pilot, not as "NEXUS on BI-Bench".
"""
import argparse
import csv as csv_mod
import glob
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "project1_agentic"))
from config import call_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sanitize import sanitize_name  # noqa: E402
from scoring import compare_with_timeout  # noqa: E402

FORBIDDEN = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|ATTACH|PRAGMA)\b", re.IGNORECASE)


def load_case_db(case_dir: str) -> tuple[sqlite3.Connection, dict[str, list[str]]]:
    """Loads every CSV in a case folder into a fresh in-memory SQLite DB, exactly like the
    paper's harness: sanitized table/column names, everything read as string dtype (BI-Bench
    tables mix currency symbols, percents, and numbers in the same column, so this avoids
    pandas mis-inferring types before the agent even sees the data)."""
    conn = sqlite3.connect(":memory:")
    schema: dict[str, list[str]] = {}
    csvs = [fp for fp in glob.glob(os.path.join(case_dir, "*.csv")) if not os.path.basename(fp).startswith("_")]
    for fp in csvs:
        try:
            df = pd.read_csv(fp, low_memory=False, dtype=str, quoting=csv_mod.QUOTE_MINIMAL, on_bad_lines="skip")
        except Exception as e:
            print(f"    skipped {os.path.basename(fp)}: {e}")
            continue
        tname = sanitize_name(os.path.basename(fp)[:-4])
        df.columns = [sanitize_name(c) for c in df.columns]
        df.to_sql(tname, conn, if_exists="replace", index=False)
        schema[tname] = list(df.columns)
    return conn, schema


def schema_prompt(schema: dict[str, list[str]]) -> str:
    lines = [f"{t}({', '.join(cols)})" for t, cols in schema.items()]
    return "\n".join(lines)


def generate_sql(question: str, schema: dict[str, list[str]], retry_error: str | None = None) -> str:
    system = (
        "You write a single SQLite SELECT statement to answer a business question, using "
        "ONLY the tables and columns listed below. Column values may be stored as text "
        "(e.g. '$1,234', '12%'); cast with CAST(REPLACE(REPLACE(col,'$',''),',','') AS REAL) "
        "as needed. Return ONLY the SQL, no markdown fences, no commentary."
    )
    prompt = f"Schema:\n{schema_prompt(schema)}\n\nQuestion: {question}\n\nSQL:"
    if retry_error:
        prompt += f"\n\nYour previous SQL failed with this error, fix it:\n{retry_error}"
    sql = call_llm(system, prompt, max_tokens=600).strip()
    return re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()


def run_case(case_id: str, question: str, case_dir: str, gt_dir: str) -> dict:
    t0 = time.time()
    conn, schema = load_case_db(case_dir)
    if not schema:
        return {"case": case_id, "status": "no_tables", "passed": False, "latency_s": 0}

    sql, error, df = None, None, None
    for attempt in range(2):  # one retry on SQL error, unlike the current sql_agent.py
        try:
            sql = generate_sql(question, schema, retry_error=error)
            if FORBIDDEN.search(sql):
                raise ValueError("forbidden statement type")
            df = pd.read_sql_query(sql, conn)
            error = None
            break
        except Exception as e:
            # pandas/SQLAlchemy echoes the entire failing query inside the exception
            # message before the real reason, so keeping the FIRST N chars (the old
            # behavior) threw away the actual error on any non-trivial query and left
            # the retry with nothing useful to fix itself with. Strip the echoed SQL
            # and keep the real message instead.
            msg = str(e)
            if sql and sql in msg:
                msg = msg.replace(sql, "<query>")
            error = msg[-400:]

    gt_files = sorted(glob.glob(os.path.join(gt_dir, f"{case_id}_*.csv")) + glob.glob(os.path.join(gt_dir, f"{case_id}.csv")))
    gt_list = []
    for gf in gt_files:
        try:
            gt_list.append(pd.read_csv(gf))
        except Exception:
            pass

    passed = False
    if df is not None and gt_list:
        passed = compare_with_timeout(gt_list, df)

    conn.close()
    return {
        "case": case_id,
        "status": "ok" if error is None else "sql_error",
        "passed": bool(passed),
        "latency_s": round(time.time() - t0, 1),
        "n_tables": len(schema),
        "n_gt_files": len(gt_files),
        "sql": sql,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing case subfolders, gt/, and queries.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--case", default=None, help="Run a single case id instead of the pilot list")
    parser.add_argument("--out", default="evaluation/bibench/pilot_results.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "queries.json") as f:
        queries = json.load(f)

    gt_dir = str(data_dir / "gt")
    case_ids = [args.case] if args.case else [
        d.name for d in data_dir.iterdir() if d.is_dir() and d.name != "gt" and d.name in queries
    ][: args.limit]

    results = []
    for i, case_id in enumerate(case_ids, 1):
        question = queries[case_id]
        print(f"[{i}/{len(case_ids)}] {case_id}: {question[:70]}...")
        r = run_case(case_id, question, str(data_dir / case_id), gt_dir)
        print(f"    -> {r['status']}, passed={r['passed']}, {r['latency_s']}s")
        results.append(r)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)

    n = len(out_df)
    n_pass = int(out_df["passed"].sum())
    print(f"\nPilot accuracy: {n_pass}/{n} = {n_pass / n * 100:.1f}%")
    print(f"SQL errors (after retry): {(out_df['status'] == 'sql_error').sum()}/{n}")
    print(f"Saved per-case results to {args.out}")
    print(
        "\nReminder: this is a size-capped pilot subset with a baseline (no data-management-"
        "tools) SQL agent, run once. Compare only against the paper's *_sql_no-tool columns "
        "in results/, and only on these same case IDs, not the paper's overall averages."
    )


if __name__ == "__main__":
    main()
