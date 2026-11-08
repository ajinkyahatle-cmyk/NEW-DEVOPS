"""Train the EstateIQ house-price model with PyCaret.

This script is the single entry point for (re)training, used both locally
(`make train`) and by the CI/CD pipeline. It:

1. loads the King County dataset and engineers the sale-date features,
2. runs a reproducible PyCaret regression experiment (tracked in MLflow),
3. enforces a quality gate (the build fails if the model is too weak),
4. saves the final pipeline to models/house_price_pipeline.pkl together with
   models/metrics.json, which the Streamlit app displays.

Usage:
    python -m src.train
    python -m src.train --min-r2 0.85 --no-mlflow
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pycaret import __version__ as pycaret_version
from pycaret.regression import RegressionExperiment

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "kc_house_data.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline"  # PyCaret adds .pkl
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
MLFLOW_DIR = BASE_DIR / "mlruns"

TARGET = "price"
SESSION_ID = 42  # fixed seed -> reproducible training in every environment
EXPERIMENT_NAME = "estateiq-house-price"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train")


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load raw data and engineer the same features used at prediction time."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    sale_date = pd.to_datetime(df["date"], format="%Y%m%dT000000")
    df["sale_year"] = sale_date.dt.year
    df["sale_month"] = sale_date.dt.month
    df["sale_day"] = sale_date.dt.day
    return df.drop(columns=["id", "date"])


def git_sha() -> str:
    """Commit the model was trained from (CI sets GITHUB_SHA)."""
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-r2",
        type=float,
        default=float(os.getenv("MIN_R2", "0.80")),
        help="Quality gate: minimum hold-out R2 (default 0.80).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking.",
    )
    args = parser.parse_args()

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_DIR.as_uri())

    log.info("Loading data from %s", DATA_PATH)
    df = load_and_prepare(DATA_PATH)
    log.info("Dataset ready: %d rows, %d columns", *df.shape)

    # ---- 1. PyCaret experiment -------------------------------------------
    exp = RegressionExperiment()
    exp.setup(
        data=df,
        target=TARGET,
        session_id=SESSION_ID,
        normalize=True,
        fold=10,
        n_jobs=-1,
        log_experiment=use_mlflow,
        experiment_name=EXPERIMENT_NAME,
        html=False,
        verbose=False,
    )

    # ---- 2. Train + cross-validate Extra Trees ---------------------------
    log.info("Training Extra Trees Regressor (10-fold cross-validation)")
    model = exp.create_model("et", verbose=False)
    cv_mean = exp.pull().loc["Mean"]

    # ---- 3. Evaluate on the untouched hold-out set -----------------------
    exp.predict_model(model, verbose=False)
    holdout = exp.pull().iloc[0]

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "pycaret_version": pycaret_version,
        "model": "ExtraTreesRegressor",
        "session_id": SESSION_ID,
        "rows": len(df),
        "cv_r2": round(float(cv_mean["R2"]), 4),
        "cv_mae": round(float(cv_mean["MAE"]), 2),
        "holdout_r2": round(float(holdout["R2"]), 4),
        "holdout_mae": round(float(holdout["MAE"]), 2),
        "holdout_rmse": round(float(holdout["RMSE"]), 2),
        "min_r2_gate": args.min_r2,
    }
    log.info("Metrics: %s", json.dumps(metrics))

    # ---- 4. Quality gate: stop the pipeline before a bad model ships -----
    if metrics["holdout_r2"] < args.min_r2:
        log.error(
            "Quality gate FAILED: hold-out R2 %.4f < %.2f",
            metrics["holdout_r2"],
            args.min_r2,
        )
        return 1
    log.info("Quality gate passed (R2 %.4f >= %.2f)", metrics["holdout_r2"], args.min_r2)

    # ---- 5. Refit on all data and save the deployable pipeline -----------
    final_model = exp.finalize_model(model)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    exp.save_model(final_model, str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Saved %s.pkl and %s", MODEL_PATH, METRICS_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())