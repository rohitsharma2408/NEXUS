# BI-Bench pilot: sampling methodology

The full BI-Bench benchmark (Hu-Chuxuan/bi-agent) has 100 cases ranging from a few KB to
702 MB of data (mean 46.6 MB, median 1.4 MB — a handful of very large cases dominate the
mean). Running all 100 through a rate-limited free-tier LLM key is impractical, so NEXUS
evaluates a **20-case pilot**, sampled as follows:

1. Restricted to the 77 cases under 25 MB (so the raw data was actually downloadable and
   the LLM's context window wasn't overwhelmed by a single oversized table).
2. Stratified by table count into three buckets — few (2-5 tables), mid (6-11), many (12+)
   — and sampled 7 / 7 / 6 cases per bucket with a fixed random seed, so the pilot isn't
   skewed toward trivially simple schemas.
3. The resulting 20 cases and their questions are listed in `pilot_case_list.csv`.

**What this pilot is not:** it is not the full BI-Bench, it does not reproduce the paper's
"with tools" mode (the join-graph and transformation library in `data_management_tools/`
are not used), and it runs the agent once per case rather than the paper's multi-sample
protocol. Report results as "NEXUS, 20-case BI-Bench pilot, SQL/no-tool baseline" — not as
an unqualified "NEXUS scores X% on BI-Bench".

**Comparable paper baseline:** the `*_sql_no-tool` columns in `results/proprietary_models.csv`
and `results/open_source_models.csv` (from the original repo) are the closest apples-to-apples
comparison, filtered to these same 20 case IDs.

## Running it

```bash
docker compose exec api python evaluation/bibench/runner.py \
    --data-dir evaluation/bibench/pilot_data --limit 20
```

Results are written to `evaluation/bibench/pilot_results.csv` (case, pass/fail, SQL used,
latency, error if any). The scoring logic in `scoring.py` is copied from the paper's own
harness (`compare_dataframes`), so a pass here means the same thing as a pass in their
`results/*.csv`.
