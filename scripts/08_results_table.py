"""Main results table: annualised Sharpe of Baseline, X-Trend, X-Trend-G and X-Trend-Q next
to the Long, TSMOM and MACD benchmarks, plus the settings and kept hyperparameters.

    python scripts/08_results_table.py                  # the four main runs
    python scripts/08_results_table.py --suffix _quick  # their --quick versions (a test)

Reads results/replication/tables/<run>_metrics.csv and <run>_training_log.csv written by
03_replicate.py. Runs that have not finished yet are skipped with a note, so it can be
run at any time. Writes to results/replication/tables/:
  main_results.csv / .md     Sharpe 1995-2023 and 2018-2023, return, drawdown, settings
  kept_hyperparameters.csv   the setting of every model kept in each window's ensemble
--max-steps-per-epoch is not saved in the logs, so pass it here to show it in the table.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "replication" / "tables"

MODELS = [  # (run name, label, loss)
    ("baseline_sharpe", "Baseline (no context)", "Sharpe"),
    ("xtrend_sharpe_cpd21_c10", "X-Trend", "Sharpe"),
    ("xtrend_gauss_cpd21_c10", "X-Trend-G", "Gaussian NLL + Sharpe"),
    ("xtrend_qre_cpd21_c10", "X-Trend-Q", "Quantile + Sharpe"),
]
BENCHMARKS = [("long", "Long"), ("tsmom", "TSMOM"), ("macd", "MACD")]
HPARAMS = ["hidden_size", "dropout", "batch_size", "max_grad_norm"]


def counts(s):
    """'64: 10, 128: 8' (value: number of kept models), most common first."""
    return ", ".join(f"{k:g}: {v}" for k, v in s.value_counts().items())


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--suffix", default="", help="run name suffix, e.g. _quick")
    p.add_argument("--max-steps-per-epoch", default="none (every sequence)",
                   help="what the runs used, for the table (not saved in the logs)")
    args = p.parse_args()

    mcfg = json.loads((ROOT / "configs" / "model.json").read_text())
    t, c = mcfg["training"], mcfg["context"]

    rows, kept_all, bench = [], [], None
    for run, label, loss in MODELS:
        name = run + args.suffix
        mpath, lpath = TABLES / f"{name}_metrics.csv", TABLES / f"{name}_training_log.csv"
        if not mpath.exists():
            print(f"skipping {label}: {mpath.relative_to(ROOT)} not found (run not finished)")
            continue
        m = pd.read_csv(mpath).set_index(["strategy", "period"])
        log = pd.read_csv(lpath)
        if bench is None:
            bench = m
        kept = log[log["kept"]]
        kept_all.append(kept.assign(model=label)[["model", "test_start", "repeat", *HPARAMS,
                                                  "val_loss", "epochs"]])
        rows.append({
            "Model": label,
            "Loss": loss,
            "Context set": "none" if run.startswith("baseline") else
                           f"CPD regimes, max {c['length']} days, {c['size']} contexts",
            "Sharpe 1995-2023": m.loc[(name, "all"), "sharpe"],
            "Sharpe 2018-2023": m.loc[(name, "2018-2023"), "sharpe"],
            "Annual return": m.loc[(name, "all"), "annual_return"],
            "Max drawdown": m.loc[(name, "all"), "max_drawdown"],
            "Ensemble size": int(log["repeat"].nunique()),
            "Settings tried per member": int(log["setting"].nunique()),
            "Mean epochs": log["epochs"].mean(),
            **{f"Kept {h}": counts(kept[h]) for h in HPARAMS},
        })
    if not rows:
        raise SystemExit("No finished runs found. Run 03_replicate.py first.")

    for key, label in BENCHMARKS:
        rows.append({"Model": label, "Loss": "-", "Context set": "-",
                     "Sharpe 1995-2023": bench.loc[(key, "all"), "sharpe"],
                     "Sharpe 2018-2023": bench.loc[(key, "2018-2023"), "sharpe"],
                     "Annual return": bench.loc[(key, "all"), "annual_return"],
                     "Max drawdown": bench.loc[(key, "all"), "max_drawdown"]})

    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "main_results.csv", index=False)
    pd.concat(kept_all).to_csv(TABLES / "kept_hyperparameters.csv", index=False)

    show = table.copy()
    for col in ["Sharpe 1995-2023", "Sharpe 2018-2023"]:
        show[col] = show[col].map("{:.2f}".format)
    for col in ["Annual return", "Max drawdown"]:
        show[col] = show[col].map("{:.1%}".format)
    show["Mean epochs"] = show["Mean epochs"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
    show = show.fillna("-")
    for col in ["Ensemble size", "Settings tried per member"]:
        show[col] = show[col].map(lambda v: v if v == "-" else int(v))

    main_cols = ["Model", "Loss", "Context set", "Sharpe 1995-2023", "Sharpe 2018-2023",
                 "Annual return", "Max drawdown"]
    hp_cols = ["Model", "Ensemble size", "Settings tried per member", "Mean epochs",
               *[f"Kept {h}" for h in HPARAMS]]
    notes = (f"Returns scaled to {mcfg['backtest']['vol_target']:.0%} annual volatility, "
             f"transaction cost {mcfg['backtest']['transaction_cost']}. Expanding windows: "
             f"train from {mcfg['backtest']['train_start'][:4]}, test in "
             f"{mcfg['backtest']['test_years']}-year blocks from "
             f"{mcfg['backtest']['first_test_start'][:4]} to {mcfg['backtest']['test_end'][:4]}. "
             f"Adam, learning rate {t['learning_rate']}, at most {t['max_epochs']} epochs, "
             f"early stopping after {t['early_stopping_patience']} epochs without a better "
             f"validation loss, batches per epoch capped at: {args.max_steps_per_epoch}. "
             f"Paper setting: ensemble of {t['n_seeds']}, {t['search_iterations']} settings "
             f"tried per member. Kept hyperparameters are value: number of kept models over "
             f"all windows; search space {mcfg['search_space']}.")
    md = (f"## Main results\n\n{to_markdown(show[main_cols])}\n\n"
          f"## Training settings and kept hyperparameters\n\n"
          f"{to_markdown(show.loc[show['Ensemble size'] != '-', hp_cols])}\n\n{notes}\n")
    (TABLES / "main_results.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"saved {(TABLES / 'main_results.csv').relative_to(ROOT)}, main_results.md and "
          f"kept_hyperparameters.csv")


if __name__ == "__main__":
    main()
