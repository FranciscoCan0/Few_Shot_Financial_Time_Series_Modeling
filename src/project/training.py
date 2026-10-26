"""Training, hyperparameter search, ensembling and out-of-sample positions for X-Trend and
the baseline (Wood et al., 2023: Section 2.3, Section 4 and Appendix C).

Settings live in configs/model.json. For one expanding window (train_window):
    repeat training.n_seeds times:
        draw training.search_iterations settings from search_space, train each with Adam
        and early stopping on the validation loss, keep the one with the best loss
    the n_seeds kept models are the ensemble; predict() averages their positions over the
    window's test period.
"""
import copy

import numpy as np
import pandas as pd
import torch

from project import models as M
from project import vectorization_batching as V

# model.json context.method -> how the target attends to the contexts in models.XTrend
CONTEXT_ATTENTION = {"cpd": "final", "final_hidden": "final", "time_equivalent": "time"}


def pick_device(device="auto"):
    """'auto' uses CUDA when available, otherwise the CPU."""
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_hparams(mcfg, rng):
    """One random draw from search_space (dropout, hidden_size, batch_size, max_grad_norm)."""
    return {k: rng.choice(v).item() for k, v in mcfg["search_space"].items()}


def _forward(model, t, zero_shot_target=False):
    return model(t["x"], None if zero_shot_target else t["ticker_id"], t["context_x"],
                 t["context_mask"], t["context_ticker_id"], context_y=t["context_y"])


@torch.no_grad()
def evaluate_loss(model, batcher, loss, batch_size, device, seed=0):
    """The loss over a whole split, pooled over every scored day (the Sharpe loss is not an
    average of per-batch values). The same seed draws the same contexts every time."""
    model.eval()
    outs, ys, masks = [], [], []
    for b in batcher.batches(batch_size, np.random.default_rng(seed), shuffle=False):
        t = V.as_torch(b, device)
        outs.append({k: v.cpu() for k, v in _forward(model, t).items() if k != "attention"})
        ys.append(t["y"].cpu())
        masks.append(t["score_mask"].cpu())
    out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    return M.model_loss(out, torch.cat(ys), loss=loss, mask=torch.cat(masks)).item()


def train_one(arrays, train, val, mcfg, kind, loss, hp, seed, device,
              max_steps_per_epoch=None, log=print):
    """Train one model with settings `hp`; return (model at its best validation loss,
    best validation loss, validation loss per epoch).

    Adam at training.learning_rate, gradient norm clipped at hp["max_grad_norm"], at most
    training.max_epochs passes over the training sequences, stopping after
    training.early_stopping_patience epochs without a better validation loss.
    max_steps_per_epoch caps the batches per epoch (None: every sequence once).
    """
    t = mcfg["training"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = M.build_model(kind, arrays["inputs"].shape[1], len(arrays["tickers"]), loss=loss,
                          context=CONTEXT_ATTENTION[mcfg["context"]["method"]],
                          hidden=hp["hidden_size"], dropout=hp["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])

    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(t["max_epochs"]):
        model.train()
        for step, b in enumerate(train.batches(hp["batch_size"], rng)):
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
            tb = V.as_torch(b, device)
            batch_loss = M.model_loss(_forward(model, tb), tb["y"], loss=loss,
                                      mask=tb["score_mask"])
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["max_grad_norm"])
            opt.step()

        v = evaluate_loss(model, val, loss, hp["batch_size"], device, seed)
        history.append(v)
        if np.isfinite(v) and v < best:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        log(f"      epoch {epoch + 1}: val loss {v:.4f}" + ("  *" if wait == 0 else ""))
        if wait >= t["early_stopping_patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, history


def train_window(arrays, window, mcfg, cfg, kind, loss, device, n_seeds=None,
                 search_iterations=None, max_steps_per_epoch=None, log=print):
    """The ensemble for one expanding window, and a log of every model trained.

    For each of n_seeds repeats: search_iterations random settings, keep the best by
    validation loss. Defaults come from model.json (paper: 10 repeats x 10 settings).
    """
    t = mcfg["training"]
    n_seeds = n_seeds or t["n_seeds"]
    search_iterations = search_iterations or t["search_iterations"]
    train = V.Batcher(arrays, window, "train", mcfg, cfg)
    val = V.Batcher(arrays, window, "val", mcfg, cfg)
    log(f"  {len(train):,} train / {len(val):,} val sequences")
    year = int(str(window["test_start"])[:4])

    ensemble, rows = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng([t["seed"], year, s])
        best, best_row = None, None
        for i in range(search_iterations):
            hp = sample_hparams(mcfg, rng)
            seed = int(rng.integers(2 ** 31))
            log(f"    repeat {s + 1}/{n_seeds}, setting {i + 1}/{search_iterations}: {hp}")
            model, v, history = train_one(arrays, train, val, mcfg, kind, loss, hp, seed,
                                          device, max_steps_per_epoch, log)
            rows.append({"test_start": str(window["test_start"]), "repeat": s, "setting": i,
                         **hp, "seed": seed, "val_loss": v, "epochs": len(history),
                         "kept": False})
            if best is None or v < best[1]:
                best, best_row = (model, v), len(rows) - 1
        ensemble.append(best[0])
        rows[best_row]["kept"] = True
    return ensemble, pd.DataFrame(rows)


@torch.no_grad()
def predict(ensemble, arrays, window, mcfg, cfg, device, batch_size=256):
    """Ensemble-averaged positions (and forecasts) for every scored day of the test period.

    Returns one row per (ticker, date): position, the forecast columns of the variant
    (mu and sigma_forecast, or q_0 ... q_12), and target, next_ret, sigma, next_date. The
    contexts are drawn once per batch and shared by every model in the ensemble.
    Zero-shot test targets get no ticker embedding (Section 3.4).
    """
    test = V.Batcher(arrays, window, "test", mcfg, cfg)
    zero_shot = mcfg["zero_shot"]["enabled"]
    rng = np.random.default_rng(mcfg["training"]["seed"])
    for m in ensemble:
        m.eval()

    parts = []
    for i in range(0, len(test), batch_size):
        idx = np.arange(i, min(i + batch_size, len(test)))
        b = test.batch(idx, rng)
        t = V.as_torch(b, device)
        outs = [_forward(m, t, zero_shot) for m in ensemble]
        avg = {k: torch.stack([o[k] for o in outs]).mean(0).cpu().numpy()
               for k in outs[0] if k != "attention"}
        rows = test.ends[idx][:, None] - test.L + 1 + np.arange(test.L)
        scored = b["score_mask"]
        rec = {"row": rows[scored], "position": avg["position"][scored]}
        if "mu" in avg:
            rec["mu"], rec["sigma_forecast"] = avg["mu"][scored], avg["sigma"][scored]
        if "quantiles" in avg:
            q = avg["quantiles"][scored]
            rec.update({f"q_{j}": q[:, j] for j in range(q.shape[1])})
        parts.append(pd.DataFrame(rec))

    pred = pd.concat(parts, ignore_index=True)
    r = pred.pop("row").to_numpy()
    meta = pd.DataFrame({"ticker": arrays["tickers"][arrays["ticker_id"][r]],
                         "date": arrays["date"][r], "next_date": arrays["next_date"][r],
                         "target": arrays["target"][r], "next_ret": arrays["next_ret"][r],
                         "sigma": arrays["sigma"][r], "row": r,
                         "test_start": str(window["test_start"])})
    return pd.concat([meta, pred], axis=1).sort_values(["ticker", "date"]).reset_index(drop=True)
