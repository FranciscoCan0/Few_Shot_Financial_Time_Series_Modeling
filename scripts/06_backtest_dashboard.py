"""Walk-forward backtest that writes its progress as it goes, for the Streamlit dashboard
(07_streamlit_dashboard.py) to watch.

    python scripts/06_backtest_dashboard.py                         # X-Trend, all 6 windows
    python scripts/06_backtest_dashboard.py --windows 3 4 5         # test 2010-2023 only
    python scripts/06_backtest_dashboard.py --model baseline --seeds 2

    streamlit run scripts/07_streamlit_dashboard.py                 # in a second terminal

For each window a fresh model (or --seeds models, averaged) is trained with early stopping
on the validation loss, and the best-validation model trades the window's test period. The
test periods are never used to pick a model. The flags pick one setting from the search
space; 03_replicate.py does the full hyperparameter search.

Writes to results/replication/live/ (Ctrl+C stops the run; finished windows are kept):
  <run>_status.json           what the run is doing now, rewritten every few seconds
  <run>.jsonl                 one line per epoch: train / val loss, val Sharpe
  <run>_predictions.parquet   out-of-sample positions of every finished window
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from project import features
from project import models as M
from project import training as T
from project import vectorization_batching as V
from project.data import ROOT

OUT = ROOT / "results" / "replication" / "live"


def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward backtest with live progress files.")
    p.add_argument("--model", choices=["xtrend", "baseline"], default="xtrend")
    p.add_argument("--loss", choices=list(M.LOSSES), default="sharpe")
    p.add_argument("--windows", type=int, nargs="+",
                   help="window numbers (0 = test 1995-2000); default all")
    p.add_argument("--seeds", type=int, default=1, help="models averaged per window")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=150,
                   help="batches per epoch (0 = every training sequence)")
    p.add_argument("--patience", type=int, help="default: training.early_stopping_patience")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--name", help="run name (default from model, loss and context)")
    return p.parse_args()


class Backtest:
    def __init__(self, args):
        self.args = args
        self.cfg = features.load_config()
        self.mcfg = copy.deepcopy(V.load_model_config(data_cfg=self.cfg))
        self.arrays = V.load_arrays(self.cfg, self.mcfg)
        self.windows = V.expanding_windows(self.mcfg)
        self.chosen = args.windows if args.windows is not None else list(range(len(self.windows)))
        self.device = T.pick_device(args.device)
        c = self.mcfg["context"]
        ctx = "" if args.model == "baseline" else f"_{c['method']}{c.get('length') or ''}_c{c['size']}"
        self.name = args.name or f"{args.model}_{args.loss}{ctx}_backtest"
        self.patience = args.patience or self.mcfg["training"]["early_stopping_patience"]

        OUT.mkdir(parents=True, exist_ok=True)
        self.log_path = OUT / f"{self.name}.jsonl"
        self.status_path = OUT / f"{self.name}_status.json"
        self.pred_path = OUT / f"{self.name}_predictions.parquet"
        self.log_path.write_text("")
        self.pred_path.unlink(missing_ok=True)
        self.preds, self.windows_done = [], []
        self.t0, self.last_status = time.time(), 0.0
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")

    def status(self, text, state="running", force=False):
        """Rewrite the status file (at most every 3 seconds unless forced)."""
        if not force and time.time() - self.last_status < 3:
            return
        self.last_status = time.time()
        s = {"name": self.name, "state": state, "status": text, "started": self.started,
             "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "minutes": round((time.time() - self.t0) / 60, 2),
             "windows": [str(self.windows[i]["test_start"])[:4] for i in self.chosen],
             "windows_done": self.windows_done, "device": str(self.device),
             "settings": vars(self.args)}
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1))
        tmp.replace(self.status_path)                # never leave a half-written file

    @torch.no_grad()
    def _val_metrics(self, model, val):
        """Validation loss (the variant's training loss) and the positions' Sharpe ratio."""
        model.eval()
        outs, ys, masks = [], [], []
        for b in val.batches(256, np.random.default_rng(0), shuffle=False):
            t = V.as_torch(b, self.device)
            outs.append({k: v.cpu() for k, v in T._forward(model, t).items() if k != "attention"})
            ys.append(t["y"].cpu())
            masks.append(t["score_mask"].cpu())
        out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
        y, mask = torch.cat(ys), torch.cat(masks)
        loss = M.model_loss(out, y, loss=self.args.loss, mask=mask).item()
        sharpe = -M.sharpe_loss(out["position"], y, mask=mask).item()
        model.train()
        return loss, sharpe

    def _train(self, year, k, train, val, seed):
        """One model with early stopping, returned at its best validation loss."""
        a = self.args
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = M.build_model(a.model, self.arrays["inputs"].shape[1],
                              len(self.arrays["tickers"]), loss=a.loss,
                              context=T.CONTEXT_ATTENTION[self.mcfg["context"]["method"]],
                              hidden=a.hidden, dropout=a.dropout).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.mcfg["training"]["learning_rate"])
        best, best_state, wait = np.inf, None, 0
        for epoch in range(1, a.epochs + 1):
            losses = []
            for step, b in enumerate(train.batches(a.batch_size, rng)):
                if a.steps_per_epoch and step >= a.steps_per_epoch:
                    break
                t = V.as_torch(b, self.device)
                loss = M.model_loss(T._forward(model, t), t["y"], loss=a.loss,
                                    mask=t["score_mask"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
                opt.step()
                losses.append(loss.item())
                self.status(f"test {year}, seed {k}: epoch {epoch}/{a.epochs}, batch {step + 1}")
            self.status(f"test {year}, seed {k}: epoch {epoch} validating", force=True)
            v, s = self._val_metrics(model, val)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"test_start": year, "seed": k, "epoch": epoch,
                                    "train_loss": float(np.mean(losses)), "val_loss": v,
                                    "val_sharpe": s,
                                    "minutes": (time.time() - self.t0) / 60}) + "\n")
            if v < best:
                best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
            if wait >= self.patience:
                break
        model.load_state_dict(best_state)
        return model

    def run(self):
        for i in self.chosen:
            w = self.windows[i]
            year = str(w["test_start"])[:4]
            self.status(f"test {year}: building batches", force=True)
            train = V.Batcher(self.arrays, w, "train", self.mcfg, self.cfg)
            val = V.Batcher(self.arrays, w, "val", self.mcfg, self.cfg)
            print(f"window test {year}: {len(train):,} train / {len(val):,} val sequences")
            ensemble = [self._train(year, k, train, val, self.args.seed + 1000 * i + k)
                        for k in range(self.args.seeds)]
            self.status(f"test {year}: trading the test period", force=True)
            self.preds.append(T.predict(ensemble, self.arrays, w, self.mcfg, self.cfg,
                                        self.device))
            pd.concat(self.preds, ignore_index=True).to_parquet(self.pred_path)
            self.windows_done.append(year)
            print(f"  done after {(time.time() - self.t0) / 60:.1f} min")


def main():
    bt = Backtest(parse_args())
    print(f"{bt.name} on {bt.device}: windows {bt.chosen}; files in "
          f"{OUT.relative_to(ROOT)}/{bt.name}*")
    try:
        bt.run()
        bt.status(f"finished {len(bt.windows_done)} windows", state="done", force=True)
    except KeyboardInterrupt:
        bt.status(f"stopped after {len(bt.windows_done)} windows", state="stopped", force=True)
        print("stopped")
    except Exception as e:
        bt.status(f"error: {e!r}", state="error", force=True)
        raise


if __name__ == "__main__":
    main()
