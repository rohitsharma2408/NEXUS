"""Stage 4 — ML Layer: trains all four models in sequence, logging each to MLflow."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train_forecasting.py",
    "train_churn.py",
    "train_anomaly.py",
    "train_supplier_risk.py",
    "validate_elasticity.py",
]

if __name__ == "__main__":
    here = Path(__file__).parent
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        result = subprocess.run([sys.executable, str(here / script)], cwd=here)
        if result.returncode != 0:
            print(f"{script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    print("\nAll ML training stages complete. Models saved to models/.")
