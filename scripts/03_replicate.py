"""Step 3: replicate X-Trend. Train an ensemble on every expanding window, trade each test
period with it, and compare against the Long, TSMOM and MACD benchmarks.

    python scripts/03_replicate.py --model xtrend --loss sharpe     # X-Trend
    python scripts/03_replicate.py --model xtrend --loss gauss      # X-Trend-G
    python scripts/03_replicate.py --model xtrend --loss qre        # X-Trend-Q
    python scripts/03_replicate.py --model baseline                 # no context set
    python scripts/03_replicate.py --quick                          # minutes: a smoke test

    python scripts/03_replicate.py --live                           # watch it in Streamlit:
    streamlit run scripts/07_streamlit_dashboard.py                 # in a second terminal

The context method, size and length, the search space and the training settings come from
configs/model.json; the flags below only shrink the run. The full paper setting trains
n_seeds x search_iterations = 100 models per window, which needs a GPU.

Writes to results/replication/:
  tables/<run>_predictions.parquet   ensemble positions (and forecasts) per test day
  tables/<run>_training_log.csv      every model trained: settings, val loss, epochs
  tables/<run>_metrics.csv           Sharpe, return, vol, max drawdown vs benchmarks
  tables/<run>_sharpe_by_year.csv    Sharpe ratio per calendar year vs benchmarks
  figures/<run>_returns.png          cumulative return, rolling Sharpe, drawdown and
                                     rolling max drawdown over time (15% vol)
  models/<run>/                      the kept models' weights per window
With --live, also results/replication/live/<run>_status.json, <run>.jsonl (one line per
epoch of every model in the search) and <run>_predictions.parquet (finished windows), the
files 07_streamlit_dashboard.py watches. Once the run finishes the dashboard shows the
tables/ version instead.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from project import evaluation as E
from project import features
from project import plotting as P
from project import training as T
from project import vectorization_batching as V
from project.data import ROOT

OUT = ROOT / "results" / "replication"
# reporting periods besides the whole sample (the paper also reports 2018-2023)
PERIODS = {"2018-2023": ("2018-01-01", None)}


def run_name(args, mcfg):
    if args.model == "baseline":
        name = f"baseline_{args.loss}"
    else:
        c = mcfg["context"]
        length = "" if c["method"] == "time_equivalent" else c["length"]
        name = f"xtrend_{args.loss}_{c['method']}{length}_c{c['size']}"
    if mcfg["zero_shot"]["enabled"]:
        name += "_zeroshot"
    return name + ("_quick" if args.quick else "")


class Live:
    """Progress files for 07_streamlit_dashboard.py, in the format 06_backtest_dashboard.py
    writes: a status file (rewritten at most every 3 seconds unless forced), one jsonl line
    per epoch, and the predictions of the finished windows."""

    def __init__(self, name, years, device, args):
        self.dir = OUT / "live"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.dir / f"{name}_status.json"
        self.log_path = self.dir / f"{name}.jsonl"
        self.pred_path = self.dir / f"{name}_predictions.parquet"
        self.log_path.write_text("")
        self.pred_path.unlink(missing_ok=True)
        self.name, self.years, self.device, self.args = name, years, device, args
        self.year, self.windows_done = None, []
        self.t0, self.last = time.time(), 0.0
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")

    def status(self, text, state="running", force=False):
        if not force and time.time() - self.last < 3:
            return
        self.last = time.time()
        s = {"name": self.name, "state": state, "status": text, "started": self.started,
             "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "minutes": round((time.time() - self.t0) / 60, 2), "windows": self.years,
             "windows_done": self.windows_done, "device": str(self.device),
             "settings": vars(self.args)}
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1))
        tmp.replace(self.status_path)                # never leave a half-written file

    def log(self, text):
        print(text)
        self.status(f"test {self.year}: {text.strip()}")

    def epoch(self, repeat, setting, **e):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"test_start": self.year, "seed": repeat, "setting": setting,
                                "model": f"repeat {repeat} setting {setting}", **e,
                                "minutes": (time.time() - self.t0) / 60}) + "\n")

    def window_done(self, preds):
        pd.concat(preds, ignore_index=True).to_parquet(self.pred_path)
        self.windows_done.append(self.year)


def main():
    parser = argparse.ArgumentParser(description="Replicate X-Trend on expanding windows.")
    parser.add_argument("--model", choices=["xtrend", "baseline"], default="xtrend")
    parser.add_argument("--loss", choices=list(T.M.LOSSES), default="sharpe",
                        help="sharpe: X-Trend / Baseline, gauss: X-Trend-G, qre: X-Trend-Q")
    parser.add_argument("--windows", type=int, nargs="+",
                        help="window numbers to run (0 = test 1995-2000); default all")
    parser.add_argument("--seeds", type=int, help="override training.n_seeds")
    parser.add_argument("--search-iterations", type=int,
                        help="override training.search_iterations")
    parser.add_argument("--max-epochs", type=int, help="override training.max_epochs")
    parser.add_argument("--max-steps-per-epoch", type=int,
                        help="cap batches per epoch (default: every training sequence)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 1 seed, 1 setting, 2 epochs of 20 batches, "
                             "first window only")
    parser.add_argument("--live", action="store_true",
                        help="write progress files for 07_streamlit_dashboard.py")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.search_iterations = args.seeds or 1, args.search_iterations or 1
        args.max_epochs = args.max_epochs or 2
        args.max_steps_per_epoch = args.max_steps_per_epoch or 20
        args.windows = args.windows or [0]

    cfg = features.load_config()
    mcfg = copy.deepcopy(V.load_model_config(data_cfg=cfg))
    if args.max_epochs:
        mcfg["training"]["max_epochs"] = args.max_epochs
    arrays = V.load_arrays(cfg, mcfg)
    windows = V.expanding_windows(mcfg)
    chosen = args.windows if args.windows is not None else range(len(windows))
    device = T.pick_device(args.device)
    name = run_name(args, mcfg)
    for d in ["tables", "figures", f"models/{name}"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    print(f"run {name} on {device}")
    live = (Live(name, [str(windows[i]["test_start"])[:4] for i in chosen], device, args)
            if args.live else None)

    preds, logs = [], []
    try:
        for i in chosen:
            w = windows[i]
            print(f"window {i}: train from {w['train_start']}, test {w['test_start']} "
                  f"to {w['test_end']}")
            t0 = time.time()
            if live:
                live.year = str(w["test_start"])[:4]
                live.status(f"test {live.year}: building batches", force=True)
            ensemble, log = T.train_window(arrays, w, mcfg, cfg, args.model, args.loss, device,
                                           args.seeds, args.search_iterations,
                                           args.max_steps_per_epoch,
                                           log=live.log if live else print,
                                           on_epoch=live.epoch if live else None)
            for k, model in enumerate(ensemble):
                torch.save(model.state_dict(),
                           OUT / "models" / name / f"test{str(w['test_start'])[:4]}_model{k}.pt")
            if live:
                live.status(f"test {live.year}: trading the test period", force=True)
            preds.append(T.predict(ensemble, arrays, w, mcfg, cfg, device))
            logs.append(log)
            if live:
                live.window_done(preds)
            print(f"  done in {(time.time() - t0) / 60:.1f} min")
    except KeyboardInterrupt:
        if live:
            live.status(f"stopped after {len(preds)} windows", state="stopped", force=True)
        raise
    except Exception as e:
        if live:
            live.status(f"error: {e!r}", state="error", force=True)
        raise

    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(OUT / "tables" / f"{name}_predictions.parquet")
    pd.concat(logs).to_csv(OUT / "tables" / f"{name}_training_log.csv", index=False)

    bench = E.benchmark_positions(arrays, pred["row"].to_numpy())
    cost = mcfg["backtest"]["transaction_cost"]
    vol = mcfg["backtest"]["vol_target"]
    daily = {name: E.portfolio_returns(pred, "position", cost, vol)}
    daily.update({b: E.portfolio_returns(bench, b, cost, vol) for b in E.BENCHMARKS})
    table = E.compare(daily, PERIODS, vol)
    table.to_csv(OUT / "tables" / f"{name}_metrics.csv")
    print("\n" + table.round(3).to_string())

    scores = E.forecast_scores(pred, T.M.QUANTILES)
    if len(scores):
        print("\nforecast quality:\n" + scores.round(4).to_string())
    P.performance_by_year(daily).to_csv(OUT / "tables" / f"{name}_sharpe_by_year.csv")
    fig = P.plot_performance(daily, title=name, path=OUT / "figures" / f"{name}_returns.png")
    plt.close(fig)
    if live:
        live.status(f"finished {len(preds)} windows", state="done", force=True)
    print(f"\nsaved results/replication/tables/{name}_*.csv/.parquet and figures/{name}_returns.png")


if __name__ == "__main__":
    main()
