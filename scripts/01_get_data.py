"""Step 1: download the futures prices from WRDS and save the raw file.

    python scripts/01_get_data.py

Uses project.data with configs/data.json (set wrds_username there first). Writes
data/raw/futures_close.parquet: one row per (ticker, date), back-adjusted prices.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project import data

if __name__ == "__main__":
    data.make_raw(data.load_config())
