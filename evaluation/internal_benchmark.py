"""
Internal evaluation — Section 19 of the blueprint: 6 levels (SQL, Analytics, Multi-table,
Prediction, Investigation, Agentic). Sample question set below; extend to 100+ as you go.
Scores each question by calling the live /investigate API and checking whether expected
row-level facts show up in the SQL evidence (a light, extensible proxy for correctness —
swap in exact-match checks once you have ground truth for each question).
"""
import argparse
import json
import time

import requests

QUESTIONS = [
    {"level": "sql", "question": "What was total revenue last month?"},
    {"level": "analytics", "question": "What is the average order value by month?"},
    {"level": "multi_table", "question": "Which product category has the highest return rate?"},
    {"level": "prediction", "question": "Which products are forecasted to sell the most units next month?"},
    {"level": "investigation", "question": "Why did revenue fall in Delhi last month?"},
    {"level": "agentic", "question": "Create an executive summary of Q3 performance including risks."},
]


def run(api_url: str):
    results = []
    for item in QUESTIONS:
        t0 = time.time()
        resp = requests.post(f"{api_url}/investigate", json={"question": item["question"]}, timeout=120)
        latency = time.time() - t0
        ok = resp.status_code == 200
        body = resp.json() if ok else {"error": resp.text}
        results.append({
            "level": item["level"],
            "question": item["question"],
            "status": resp.status_code,
            "latency_sec": round(latency, 2),
            "confidence": body.get("confidence"),
            "answer_preview": (body.get("answer") or "")[:200],
        })
        print(f"[{item['level']}] {item['question']} -> {resp.status_code} ({latency:.1f}s)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--out", default="evaluation/internal_benchmark_results.json")
    args = parser.parse_args()

    results = run(args.api_url)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
