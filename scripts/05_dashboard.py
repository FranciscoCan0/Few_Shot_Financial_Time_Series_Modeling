"""Live training dashboard: train one model on one expanding window and watch it learn.

    python scripts/05_dashboard.py                                  # X-Trend, test 2005-2010
    python scripts/05_dashboard.py --loss qre --window 4 --epochs 30
    python scripts/05_dashboard.py --no-window                      # no GUI: refresh a PNG

A window opens and redraws every --refresh seconds while the model trains in the background:
  batch loss          the training loss of every mini-batch (and a moving average)
  epoch loss          mean training loss vs validation loss per epoch (lower is better)
  validation Sharpe   Sharpe ratio of the model's positions on the validation split
  test panels         the current model on the test period vs Long / TSMOM / MACD:
                      cumulative return, rolling Sharpe, drawdown (all at 15% vol)
The test panels are for watching only: the model is chosen on validation loss, never on
the test period. Epoch 0 is the untrained model.

Data, context set and training settings come from configs/model.json; the flags pick one
setting from the search space. Closing the window stops training. Writes to
results/replication/live/:
  <run>.jsonl          one line of numbers per epoch
  <run>_dashboard.png  the dashboard at the end
"""
import argparse
import copy
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np
import torch

from project import evaluation as E
from project import features
from project import models as M
from project import plotting as P
from project import training as T
from project import vectorization_batching as V
from project.data import ROOT

OUT = ROOT / "results" / "replication" / "live"


def parse_args():
    p = argparse.ArgumentParser(description="Live dashboard for one training run.")
    p.add_argument("--model", choices=["xtrend", "baseline"], default="xtrend")
    p.add_argument("--loss", choices=list(M.LOSSES), default="sharpe")
    p.add_argument("--window", type=int, default=2,
                   help="expanding window number (0 = test 1995-2000, 2 = test 2005-2010)")
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
    p.add_argument("--refresh", type=float, default=3.0, help="seconds between redraws")
    p.add_argument("--no-window", action="store_true",
                   help="no GUI: rewrite the PNG every refresh instead")
    return p.parse_args()


class Run:
    """Trains in a background thread and keeps everything the dashboard draws."""

    def __init__(self, args):
        self.args = args
        self.cfg = features.load_config()
        self.mcfg = copy.deepcopy(V.load_model_config(data_cfg=self.cfg))
        self.arrays = V.load_arrays(self.cfg, self.mcfg)
        self.window = V.expanding_windows(self.mcfg)[args.window]
        self.device = T.pick_device(args.device)
        self.train = V.Batcher(self.arrays, self.window, "train", self.mcfg, self.cfg)
        self.val = V.Batcher(self.arrays, self.window, "val", self.mcfg, self.cfg)
        c = self.mcfg["context"]
        ctx = "" if args.model == "baseline" else f"_{c['method']}{c.get('length') or ''}_c{c['size']}"
        self.name = f"{args.model}_{args.loss}{ctx}_test{str(self.window['test_start'])[:4]}"
        self.patience = args.patience or self.mcfg["training"]["early_stopping_patience"]

        self.lock, self.stop = threading.Lock(), threading.Event()
        self.batch_loss, self.epochs = [], []       # epochs: one dict per epoch
        self.test_daily, self.bench_daily = None, {}
        self.status, self.done, self.error = "starting", False, None
        self.t0 = time.time()
        OUT.mkdir(parents=True, exist_ok=True)
        self.log_path = OUT / f"{self.name}.jsonl"
        self.log_path.write_text("")

    @torch.no_grad()
    def _val_metrics(self, model):
        """Validation loss (the variant's training loss) and the positions' Sharpe ratio."""
        model.eval()
        outs, ys, masks = [], [], []
        for b in self.val.batches(256, np.random.default_rng(0), shuffle=False):
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

    def _test(self, model):
        pred = T.predict([model], self.arrays, self.window, self.mcfg, self.cfg, self.device)
        model.train()
        if not self.bench_daily:
            bench = E.benchmark_positions(self.arrays, pred["row"].to_numpy())
            self.bench_daily = {b: E.portfolio_returns(bench, b) for b in E.BENCHMARKS}
        return E.portfolio_returns(pred)

    def _record(self, epoch, train_loss, model):
        val_loss, val_sharpe = self._val_metrics(model)
        daily = self._test(model)
        row = {"epoch": epoch, "train_loss": None if np.isnan(train_loss) else train_loss,
               "val_loss": val_loss,
               "val_sharpe": val_sharpe, "test_sharpe": float(E.metrics(daily)["sharpe"]),
               "minutes": (time.time() - self.t0) / 60}
        with self.lock:
            self.epochs.append(row)
            self.test_daily = daily
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return val_loss

    def work(self):
        try:
            a = self.args
            torch.manual_seed(a.seed)
            rng = np.random.default_rng(a.seed)
            model = M.build_model(a.model, self.arrays["inputs"].shape[1],
                                  len(self.arrays["tickers"]), loss=a.loss,
                                  context=T.CONTEXT_ATTENTION[self.mcfg["context"]["method"]],
                                  hidden=a.hidden, dropout=a.dropout).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.mcfg["training"]["learning_rate"])
            self.status = "epoch 0: scoring the untrained model"
            best, wait = self._record(0, np.nan, model), 0
            for epoch in range(1, a.epochs + 1):
                losses = []
                for step, b in enumerate(self.train.batches(a.batch_size, rng)):
                    if self.stop.is_set():
                        self.status = "stopped"
                        return
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
                    with self.lock:
                        self.batch_loss.append(loss.item())
                    self.status = f"epoch {epoch}/{a.epochs}, batch {step + 1}"
                self.status = f"epoch {epoch}: validating"
                v = self._record(epoch, float(np.mean(losses)), model)
                if v < best:
                    best, wait = v, 0
                else:
                    wait += 1
                if wait >= self.patience:
                    self.status = f"early stop after epoch {epoch} (no better val loss in {wait})"
                    return
            self.status = f"finished {a.epochs} epochs"
        except Exception as e:                       # show the error on the dashboard
            self.error, self.status = repr(e), f"error: {e!r}"
            raise
        finally:
            self.done = True

    # ---------------------------------------------------------------- drawing
    def draw(self, fig, axes):
        with self.lock:
            batch = np.array(self.batch_loss)
            epochs = list(self.epochs)
            test = self.test_daily
        for ax in axes.flat:
            ax.clear()
            ax.grid(alpha=0.3)
        a1, a2, a3, a4, a5, a6 = axes.flat

        if len(batch):
            a1.plot(batch, color="lightgray", lw=0.7)
            k = max(1, min(50, len(batch) // 5))
            a1.plot(np.arange(k - 1, len(batch)), np.convolve(batch, np.ones(k) / k, "valid"),
                    color="tab:blue", label=f"{k}-batch average")
            a1.legend(fontsize=8)
        a1.set_title("training loss per batch")
        a1.set_xlabel("batch")

        if epochs:
            ep = [r["epoch"] for r in epochs]
            a2.plot(ep[1:], [r["train_loss"] for r in epochs][1:], marker="o", ms=3, label="train")
            a2.plot(ep, [r["val_loss"] for r in epochs], marker="o", ms=3, label="val")
            best = int(np.argmin([r["val_loss"] for r in epochs]))
            a2.scatter(ep[best], epochs[best]["val_loss"], s=90, facecolors="none",
                       edgecolors="k", zorder=3, label=f"best val (epoch {ep[best]})")
            a2.legend(fontsize=8)
            a3.plot(ep, [r["val_sharpe"] for r in epochs], marker="o", ms=3, label="validation")
            a3.plot(ep, [r["test_sharpe"] for r in epochs], marker="o", ms=3, ls="--",
                    label="test (watch only)")
            a3.axhline(0, color="k", lw=0.6)
            a3.legend(fontsize=8)
        a2.set_title("loss per epoch (lower is better)")
        a2.set_xlabel("epoch")
        a3.set_title("Sharpe ratio per epoch")
        a3.set_xlabel("epoch")

        if test is not None:
            series = {"model": test, **self.bench_daily}
            for name, d in series.items():
                d = E.rescale(d.dropna())
                style = {"lw": 2 if name == "model" else 1}
                m = E.metrics(d)
                a4.plot(d.index, d.cumsum(), label=f"{name} ({m['sharpe']:.2f})", **style)
                a5.plot(d.index, P.rolling_sharpe(d, 126), **style)
                line, = a6.plot(d.index, E.drawdown(d), **style)
                if name == "model":
                    a6.plot(d.index, P.max_drawdown_to_date(d), ls="--", lw=1,
                            color=line.get_color(), label="model: max drawdown to date")
            a4.legend(fontsize=8)
            a5.axhline(0, color="k", lw=0.6)
            a6.legend(fontsize=8, loc="lower left")
            for ax in (a4, a5, a6):
                ax.tick_params(axis="x", labelrotation=30)
        a4.set_title("test: cumulative return, 15% vol (Sharpe)")
        a5.set_title("test: rolling 126-day Sharpe")
        a6.set_title("test: drawdown")

        best = (f", best val Sharpe {max(r['val_sharpe'] for r in epochs):.2f}" if epochs else "")
        fig.suptitle(f"{self.name}   |   {self.status}   |   "
                     f"{(time.time() - self.t0) / 60:.1f} min{best}", fontsize=11)
        fig.tight_layout()


def main():
    args = parse_args()
    if args.no_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    run = Run(args)
    print(f"{run.name} on {run.device}: {len(run.train):,} train / {len(run.val):,} val "
          f"sequences; log {run.log_path.relative_to(ROOT)}")
    worker = threading.Thread(target=run.work, daemon=True)
    worker.start()

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    png = OUT / f"{run.name}_dashboard.png"
    if args.no_window:
        while not run.done:
            time.sleep(args.refresh)
            run.draw(fig, axes)
            fig.savefig(png, dpi=100)
    else:
        def update(_):
            run.draw(fig, axes)
            if run.done:
                anim.event_source.stop()
        anim = FuncAnimation(fig, update, interval=args.refresh * 1000,
                             cache_frame_data=False)
        plt.show()                                   # blocks until the window is closed
        run.stop.set()
        worker.join(timeout=60)
    run.draw(fig, axes)
    fig.savefig(png, dpi=100)
    print(f"{run.status}; saved {png.relative_to(ROOT)}")
    if run.error:
        raise SystemExit(run.error)


if __name__ == "__main__":
    main()
