"""
Generate data and train all ML models for local or Streamlit Cloud deploy.

Run once from the project root before starting the app (or before pushing
to GitHub if you plan to force-add gitignored artifacts for Community Cloud).

Usage (Windows):
    py scripts/bootstrap_artifacts.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STEPS: list[tuple[str, Path]] = [
    ("Generate acquisition and transaction CSVs", PROJECT_ROOT / "src" / "data_generation.py"),
    ("Train retention classifier", PROJECT_ROOT / "src" / "train_retention_model.py"),
    ("Train LTV regressor", PROJECT_ROOT / "src" / "train_ltv_model.py"),
    ("Train BG/NBD + Gamma-Gamma model", PROJECT_ROOT / "src" / "train_bgnbd_model.py"),
]


def main() -> None:
    """Run data generation and all training scripts in order."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    for label, script in STEPS:
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        logger.info("=== %s ===", label)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Step failed ({label}): exit code {result.returncode}")

    logger.info("Bootstrap complete. Artifacts are in data/ and models/.")


if __name__ == "__main__":
    main()
