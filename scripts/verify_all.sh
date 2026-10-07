#!/usr/bin/env bash
# One command to run the whole NEXUS verification on the REAL warehouse, inside the api container.
# Usage:  bash scripts/verify_all.sh
# Writes everything to verify_report.txt in the project folder and prints a PASS/FAIL summary.
set -u
cd "$(dirname "$0")/.."
REPORT=verify_report.txt
: > "$REPORT"

docker compose up -d --build api >/dev/null 2>&1 || { echo "build failed - run: docker compose up -d --build api"; exit 1; }
sleep 8

declare -a NAMES=() CODES=()
step () {  # step "label" command...
  local name="$1"; shift
  echo -e "\n\n########## $name ##########" | tee -a "$REPORT"
  docker compose exec -T api "$@" 2>&1 | tee -a "$REPORT"
  local rc=${PIPESTATUS[0]}   # must be read immediately: any later command resets PIPESTATUS
  NAMES+=("$name"); CODES+=("$rc")
}

step "ML training (all stages)"        python project2_analytics/ml/train_all.py
step "Elasticity validation"           python project2_analytics/ml/validate_elasticity.py
step "Supplier risk diagnostics"       python project2_analytics/ml/train_supplier_risk.py
step "Forecast backtest"               python evaluation/forecast_backtest.py
step "Forecast prediction intervals"    python evaluation/forecast_intervals.py
step "Anomaly evaluation"              python evaluation/anomaly_eval.py
step "Full test suite (real data)"     pytest project1_agentic/tests project2_analytics/tests evaluation/tests -q
step "MLflow experiments persisted"    ls /app/mlflow_data/mlruns

echo -e "\n\n==================== SUMMARY ====================" | tee -a "$REPORT"
for i in "${!NAMES[@]}"; do
  [ "${CODES[$i]}" -eq 0 ] && s=PASS || s="FAIL (exit ${CODES[$i]})"
  printf "%-36s %s\n" "${NAMES[$i]}" "$s" | tee -a "$REPORT"
done
echo -e "\nFull output saved to: $REPORT"
