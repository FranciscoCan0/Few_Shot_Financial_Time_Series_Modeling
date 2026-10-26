"""Step 2: build everything the models train on from the raw file.

    python scripts/02_build_dataset.py                  # features, CPD (~3 h), arrays
    python scripts/02_build_dataset.py --skip-cpd       # features + arrays, keep saved CPD
    python scripts/02_build_dataset.py --regimes-only   # re-cut saved CPD scores + arrays

  features  project.features.make_processed  -> data/processed/xtrend_features.parquet
  CPD       project.features.make_cpd        -> data/processed/cpd_scores.parquet,
                                                cpd_regimes.parquet
  arrays    project.vectorization_batching   -> data/processed/xtrend_arrays.npz
Settings: configs/data.json (features, CPD) and configs/model.json (arrays path).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project import features
from project import vectorization_batching as V

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-cpd", action="store_true",
                       help="build features and arrays only, reusing the saved CPD regimes")
    group.add_argument("--regimes-only", action="store_true",
                       help="re-segment the saved CPD scores (seconds), then rebuild arrays")
    args = parser.parse_args()

    cfg = features.load_config()
    mcfg = V.load_model_config(data_cfg=cfg)
    if not args.regimes_only:
        features.make_processed(cfg)
    if args.regimes_only:
        features.make_cpd(cfg, recompute_scores=False)
    elif not args.skip_cpd:
        features.make_cpd(cfg)
    V.make_arrays(cfg, mcfg)
