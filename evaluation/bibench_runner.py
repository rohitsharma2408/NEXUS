"""
BI-Bench runner — external validation of the AI analyst (Section 5, blueprint).

This does NOT reimplement BI-Agent; it clones github.com/Hu-Chuxuan/bi-agent, reads its
queries.json, and scores NEXUS's own /investigate endpoint against each case's
natural-language question and ground truth, for an apples-to-apples comparison with the
paper's baseline/tool-augmented numbers.

Usage:
    git clone https://github.com/Hu-Chuxuan/bi-agent.git external/bi-agent
    python evaluation/bibench_runner.py --bibench-dir external/bi-agent/bi-bench --api-url http://localhost:8000

Remember (Section 5.2 scope note): a good score here proves general BI navigation skill on
unfamiliar schemas — it does NOT prove the agents understand the Global E-Commerce company.
Report the two evaluations separately.
"""
import argparse
import json
from pathlib import Path

import requests


def load_cases(bibench_dir: Path):
    queries_path = bibench_dir / "queries.json"
    with open(queries_path) as f:
        return json.load(f)


def score_case(case_id: str, question: str, gt_dir: Path, api_url: str) -> dict:
    resp = requests.post(f"{api_url}/investigate", json={"question": question}, timeout=180)
    answer = resp.json().get("answer", "") if resp.status_code == 200 else ""

    gt_files = list(gt_dir.glob("*.csv")) if gt_dir.exists() else []
    # Placeholder scoring: presence of the ground-truth file's numeric values in the answer
    # text. Swap for BI-Agent's own SQL-execution-accuracy comparator for a rigorous score.
    hits = 0
    for gt_file in gt_files:
        content = gt_file.read_text()
        tokens = [t for t in content.replace(",", " ").split() if t.replace(".", "", 1).isdigit()]
        hits += sum(1 for t in tokens if t in answer)

    return {
        "case_id": case_id,
        "question": question,
        "status": resp.status_code,
        "answer_preview": answer[:200],
        "gt_numeric_hits": hits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bibench-dir", required=True, help="Path to the cloned bi-agent/bi-bench directory")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=20, help="Number of cases to run (BI-Bench is large)")
    parser.add_argument("--out", default="evaluation/bibench_results.json")
    args = parser.parse_args()

    bibench_dir = Path(args.bibench_dir)
    cases = load_cases(bibench_dir)

    results = []
    for i, (case_id, question) in enumerate(cases.items()):
        if i >= args.limit:
            break
        gt_dir = bibench_dir / case_id / "gt"
        result = score_case(case_id, question, gt_dir, args.api_url)
        results.append(result)
        print(f"[{case_id}] {result['status']} — {result['gt_numeric_hits']} GT numeric hits")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} case results to {args.out}")
    print(
        "Reminder: this score measures generalization to unfamiliar BI schemas, not "
        "understanding of the Global E-Commerce company — report it separately from the "
        "internal benchmark (see evaluation/internal_benchmark.py)."
    )


if __name__ == "__main__":
    main()
