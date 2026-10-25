"""X-Trend training data as arrays, and mini-batches with a context set per target
(Wood et al., 2023: Sections 2.1, 2.3, 3.2 and 4).

Settings live in configs/model.json (data settings stay in configs/data.json). Run as a
script to build the arrays file and print the size of every split:
    python -m project.vectorization_batching

  1. make_arrays / load_arrays: the processed features and CPD regimes as flat numpy
     arrays, one row per (ticker, date) sorted by ticker then date. Saved once to
     arrays_path and rebuilt only when the processed files change.
  2. expanding_windows: train 1990-1995 / test 1995-2000, train 1990-2000 / test 2000-2005, ...
  3. Batcher(arrays, window, split): the target sequences of one split ("train", "val" or
     "test"). Each batch draws a fresh context set for every target by array indexing.

A row belongs to the period that contains its NEXT trading date, because its target is the
next day's return. Example: the last row of 1994 predicts the first return of 1995, so it
is a test row of the 1995-2000 window. No target is used in two periods.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project import features
from project.data import ROOT, _is_int, bad_config

MODEL_CONFIG_PATH = ROOT / "configs" / "model.json"
REQUIRED_KEYS = ["arrays_path", "sequences", "context", "architecture", "training",
                 "search_space", "losses", "backtest", "zero_shot"]
CONTEXT_METHODS = ["final_hidden", "time_equivalent", "cpd"]
SPLITS = ["train", "val", "test"]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_model_config(path=MODEL_CONFIG_PATH, data_cfg=None):
    with open(path) as f:
        mcfg = json.load(f)
    validate_model_config(mcfg, data_cfg or features.load_config())
    return mcfg


def validate_model_config(mcfg, cfg):
    """Raise ValueError listing every problem in model.json (checked against data.json)."""
    bad_config([f"missing key '{k}'" for k in REQUIRED_KEYS if k not in mcfg])
    errors = []
    if Path(mcfg["arrays_path"]).is_absolute():
        errors.append("arrays_path must be relative to the project root")

    s = mcfg["sequences"]
    for k, lo in [("target_length", 2), ("warmup_length", 0), ("train_stride", 1),
                  ("test_stride", 1)]:
        if not _is_int(s.get(k)) or s[k] < lo:
            errors.append(f"sequences.{k} must be an integer >= {lo}, got {s.get(k)!r}")
    if not errors:
        if s["warmup_length"] >= s["target_length"]:
            errors.append("sequences.warmup_length must be smaller than target_length")
        if s["target_length"] != cfg["contexts"]["target_length"]:
            errors.append("sequences.target_length must equal contexts.target_length in "
                          "data.json (the time-equivalent context length)")
        if s["test_stride"] > s["target_length"] - s["warmup_length"]:
            errors.append("sequences.test_stride must be <= target_length - warmup_length, "
                          "or some test days are never scored")

    c, x = mcfg["context"], cfg["contexts"]
    if c.get("method") not in CONTEXT_METHODS:
        errors.append(f"context.method must be one of {CONTEXT_METHODS}, got {c.get('method')!r}")
    if c.get("size") not in x["sizes"]:
        errors.append(f"context.size must be one of data.json contexts.sizes {x['sizes']}, "
                      f"got {c.get('size')!r}")
    allowed = {"final_hidden": x["final_hidden_lengths"], "cpd": x["cpd_max_lengths"]}
    if c.get("method") in allowed and c.get("length") not in allowed[c["method"]]:
        errors.append(f"context.length for {c['method']} must be one of "
                      f"{allowed[c['method']]}, got {c.get('length')!r}")
    if not isinstance(c.get("anytime_in_training"), bool):
        errors.append("context.anytime_in_training must be true or false")

    t = mcfg["training"]
    for k in ["max_epochs", "early_stopping_patience", "search_iterations", "n_seeds"]:
        if not _is_int(t.get(k)) or t[k] < 1:
            errors.append(f"training.{k} must be a positive integer, got {t.get(k)!r}")
    if not _is_int(t.get("seed")):
        errors.append(f"training.seed must be an integer, got {t.get('seed')!r}")
    if not _num(t.get("learning_rate")) or t["learning_rate"] <= 0:
        errors.append("training.learning_rate must be a positive number")
    if not _num(t.get("validation_fraction")) or not 0 < t["validation_fraction"] < 1:
        errors.append("training.validation_fraction must be between 0 and 1")

    for k, v in mcfg["search_space"].items():
        if not isinstance(v, list) or not v or not all(_num(i) and i > 0 for i in v):
            errors.append(f"search_space.{k} must be a non-empty list of positive numbers")
    for k in ["dropout", "hidden_size", "batch_size", "max_grad_norm"]:
        if k not in mcfg["search_space"]:
            errors.append(f"missing key 'search_space.{k}'")

    loss = mcfg["losses"]
    for k in ["gaussian", "quantile"]:
        if not _num(loss.get(k, {}).get("alpha")) or loss[k]["alpha"] < 0:
            errors.append(f"losses.{k}.alpha must be a non-negative number")
    q = loss.get("quantile", {}).get("quantiles")
    if not isinstance(q, list) or not q or not all(_num(v) and 0 < v < 1 for v in q) \
            or q != sorted(set(q)):
        errors.append("losses.quantile.quantiles must be increasing numbers between 0 and 1")

    b = mcfg["backtest"]
    try:
        start, first, end = (pd.Timestamp(b[k]) for k in
                             ["train_start", "first_test_start", "test_end"])
        if not start < first < end:
            errors.append("backtest dates must satisfy train_start < first_test_start < test_end")
        if start < pd.Timestamp(cfg["features"]["sample_start"]):
            errors.append("backtest.train_start must not be before features.sample_start")
        if end > pd.Timestamp(cfg["end_date"]):
            errors.append("backtest.test_end must not be after end_date in data.json")
    except (KeyError, ValueError) as e:
        errors.append(f"bad backtest date: {e}")
    if not _is_int(b.get("test_years")) or b["test_years"] < 1:
        errors.append("backtest.test_years must be a positive integer")

    z, universe = mcfg["zero_shot"], set(cfg["universe"])
    if not isinstance(z.get("enabled"), bool):
        errors.append("zero_shot.enabled must be true or false")
    if not _is_int(z.get("n_test_assets")) or not 1 <= z["n_test_assets"] < len(universe):
        errors.append(f"zero_shot.n_test_assets must be between 1 and {len(universe) - 1}")
    elif z.get("test_assets"):
        unknown = sorted(set(z["test_assets"]) - universe)
        if unknown:
            errors.append(f"zero_shot.test_assets not in the universe: {unknown}")
        if len(set(z["test_assets"])) != z["n_test_assets"]:
            errors.append("zero_shot.test_assets must list exactly n_test_assets tickers "
                          "(or be empty to draw them with zero_shot.seed)")
    bad_config(errors)


# --------------------------------------------------------------------------
# arrays
# --------------------------------------------------------------------------

def make_arrays(cfg=None, mcfg=None):
    """Flatten the processed features (and CPD regimes) into arrays and save them.

    inputs (rows, 8) float32       the model inputs, in features.feature_columns order
    target, next_ret, sigma (rows) float32
    date, next_date (rows)         datetime64[D]; next_date = the date `target` is earned
    ticker_id, offset (rows)       offset = row number within its ticker (0, 1, 2, ...)
    tickers (49)                   ticker name for each ticker_id
    regimes_<L>_start/_end         row positions of each CPD regime, at most L rows long
    """
    cfg = cfg or features.load_config()
    mcfg = mcfg or load_model_config(data_cfg=cfg)
    panel = (pd.read_parquet(ROOT / cfg["processed_path"])
               .sort_values(["ticker", "date"]).reset_index(drop=True))
    raw = pd.read_parquet(ROOT / cfg["raw_path"], columns=["ticker", "date"])
    raw = raw.sort_values(["ticker", "date"])
    raw["next_date"] = raw.groupby("ticker")["date"].shift(-1)
    panel = panel.merge(raw, on=["ticker", "date"], how="left", validate="1:1")
    if panel["next_date"].isna().any():
        raise ValueError("processed rows without a next raw date: rebuild the processed file")

    out = {
        "inputs": panel[features.feature_columns(cfg)].to_numpy(np.float32),
        "target": panel["target"].to_numpy(np.float32),
        "next_ret": panel["next_ret"].to_numpy(np.float32),
        "sigma": panel["sigma"].to_numpy(np.float32),
        "date": panel["date"].to_numpy("datetime64[D]"),
        "next_date": panel["next_date"].to_numpy("datetime64[D]"),
        "ticker_id": panel["ticker_id"].to_numpy(np.int64),
        "offset": panel.groupby("ticker").cumcount().to_numpy(np.int64),
        "tickers": panel.groupby("ticker_id")["ticker"].first().to_numpy(str),
        "input_columns": np.array(features.feature_columns(cfg)),
    }

    regimes_path = ROOT / cfg["regimes_path"]
    if regimes_path.exists():
        regimes = pd.read_parquet(regimes_path)
        pos = pd.Series(np.arange(len(panel)),
                        index=pd.MultiIndex.from_frame(panel[["ticker", "date"]]))
        for L, g in regimes.groupby("max_length"):
            s = pos.reindex(pd.MultiIndex.from_arrays([g["ticker"], g["start_date"]])).to_numpy()
            e = pos.reindex(pd.MultiIndex.from_arrays([g["ticker"], g["end_date"]])).to_numpy()
            ok = ~np.isnan(s) & ~np.isnan(e)              # inside the feature sample
            s, e = s[ok].astype(np.int64), e[ok].astype(np.int64)
            # Algorithm 1 can return max_length + 1 days: keep the most recent max_length
            out[f"regimes_{L}_start"], out[f"regimes_{L}_end"] = np.maximum(s, e - L + 1), e

    path = ROOT / mcfg["arrays_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"saved {path.relative_to(ROOT)}: {len(panel):,} rows, "
          f"{len(out['tickers'])} tickers, {out['inputs'].shape[1]} inputs")
    return out


def load_arrays(cfg=None, mcfg=None, rebuild=False):
    """The arrays file as a dict, rebuilt first if missing or older than its sources."""
    cfg = cfg or features.load_config()
    mcfg = mcfg or load_model_config(data_cfg=cfg)
    path = ROOT / mcfg["arrays_path"]
    sources = [ROOT / cfg[k] for k in ["processed_path", "regimes_path", "raw_path"]]
    newest = max(p.stat().st_mtime for p in sources if p.exists())
    if rebuild or not path.exists() or path.stat().st_mtime < newest:
        return make_arrays(cfg, mcfg)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# --------------------------------------------------------------------------
# periods
# --------------------------------------------------------------------------

def expanding_windows(mcfg):
    """Train on [train_start, test_start), test on [test_start, test_end); dates are the
    days the targets are earned. The last test window is cut at backtest.test_end."""
    b = mcfg["backtest"]
    start = np.datetime64(b["train_start"], "D")
    end = np.datetime64(b["test_end"], "D") + 1                 # exclusive
    t, windows = pd.Timestamp(b["first_test_start"]), []
    while np.datetime64(t, "D") < end:
        t2 = min(np.datetime64(t + pd.DateOffset(years=b["test_years"]), "D"), end)
        windows.append({"train_start": start, "test_start": np.datetime64(t, "D"),
                        "test_end": t2})
        t = pd.Timestamp(t2)
    return windows


def zero_shot_assets(mcfg, tickers):
    """The tickers held out for zero-shot testing, or None when zero-shot is off."""
    z = mcfg["zero_shot"]
    if not z["enabled"]:
        return None
    if z["test_assets"]:
        return sorted(z["test_assets"])
    rng = np.random.default_rng(z["seed"])
    return sorted(rng.choice(sorted(tickers), z["n_test_assets"], replace=False).tolist())


# --------------------------------------------------------------------------
# batches
# --------------------------------------------------------------------------

def _duplicates(k):
    """True where a value repeats an earlier one in the same row."""
    order = np.argsort(k, axis=1, kind="stable")
    s = np.take_along_axis(k, order, 1)
    dup_sorted = np.zeros_like(s, bool)
    dup_sorted[:, 1:] = s[:, 1:] == s[:, :-1]
    dup = np.zeros_like(dup_sorted)
    np.put_along_axis(dup, order, dup_sorted, 1)
    return dup


def _sample_distinct(n, size, rng, tries=20):
    """For each row i, `size` distinct integers from 0..n[i]-1 (needs n[i] >= size)."""
    k = (rng.random((len(n), size)) * n[:, None]).astype(np.int64)
    for _ in range(tries):
        dup = _duplicates(k)
        if not dup.any():
            return k
        k[dup] = (rng.random(dup.sum()) * np.broadcast_to(n[:, None], k.shape)[dup]).astype(np.int64)
    for i in np.flatnonzero(_duplicates(k).any(axis=1)):   # rare: pool barely > size
        k[i] = rng.choice(n[i], size, replace=False)
    return k


class Batcher:
    """Target sequences of one split of one expanding window, plus their context pool.

    Target sequences: `target_length` consecutive rows of one ticker. A position is scored
    (score_mask) when it is past the first `warmup_length` positions and its row belongs to
    this split. train and val use `train_stride` between sequence ends; test tiles each
    ticker's test period backwards from its last day every `test_stride` rows, so with
    stride = target_length - warmup_length every test day is scored exactly once.

    Splits (rows assigned by the date their target is earned):
      train  the first 90% of each asset's rows in [train_start, test_start)
      val    the last 10% of them (training.validation_fraction)
      test   rows in [test_start, test_end)
    Zero-shot: the held-out tickers are test targets only, and never contexts.

    Contexts: a fresh set of `context.size` per target, drawn without replacement.
      train/val contexts come from train rows only; test contexts from any row.
      Every test context ends before its target's first scored day: its last target
      (the next-day return) is known by then. In train/val the same rule applies unless
      context.anytime_in_training is true (the paper's setting: any train row).
    Targets with fewer than `size` possible contexts are dropped (counted in `dropped`).
    """

    def __init__(self, arrays, window, split, mcfg, cfg=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.a, self.split = arrays, split
        s, c = mcfg["sequences"], mcfg["context"]
        self.L, self.warmup, self.size = s["target_length"], s["warmup_length"], c["size"]
        self.Lc = self.L if c["method"] == "time_equivalent" else c["length"]

        a, nd, tid = arrays, arrays["next_date"], arrays["ticker_id"]
        history = (nd >= window["train_start"]) & (nd < window["test_start"])
        val = self._last_fraction(history, mcfg["training"]["validation_fraction"])
        train = history & ~val
        held_out = zero_shot_assets(mcfg, a["tickers"])
        unseen = (np.isin(a["tickers"][tid], held_out) if held_out
                  else np.zeros(len(nd), bool))

        if split == "test":
            rows = (nd >= window["test_start"]) & (nd < window["test_end"])
            rows &= unseen if held_out else True
            ctx_rows = (nd < window["test_end"]) & ~unseen
        else:
            rows = (train if split == "train" else val) & ~unseen
            ctx_rows = train & ~unseen
        self.score_rows = rows
        ends = self._sequence_ends(rows, s["test_stride"] if split == "test"
                                   else s["train_stride"])
        self.unscored = int(rows.sum() - self._scored(ends).sum())

        # context pool sorted by the date its last target is earned
        cs, ce = self._context_windows(ctx_rows, c["method"])
        order = np.argsort(nd[ce], kind="stable")
        self.ctx_start, self.ctx_end, self.ctx_known = cs[order], ce[order], nd[ce][order]

        if split != "test" and c["anytime_in_training"]:
            n_avail = np.full(len(ends), len(ce))
        else:
            first_scored = a["date"][ends - self.L + 1 + self.warmup]
            n_avail = np.searchsorted(self.ctx_known, first_scored, side="right")
        keep = n_avail >= self.size
        self.dropped = int((~keep).sum())
        self.ends, self.n_avail = ends[keep], n_avail[keep]

    def _last_fraction(self, rows, frac):
        """The last `frac` of each ticker's rows in `rows`."""
        out = np.zeros_like(rows)
        tid = self.a["ticker_id"]
        for t in np.unique(tid[rows]):
            r = np.flatnonzero(rows & (tid == t))
            out[r[len(r) - int(np.ceil(frac * len(r))):]] = True
        return out

    def _sequence_ends(self, rows, stride):
        """Last row of each target sequence, every `stride` rows back from each ticker's
        last row in `rows`; sequences need `target_length` rows of history."""
        tid, ends = self.a["ticker_id"], []
        for t in np.unique(tid[rows]):
            r = np.flatnonzero(rows & (tid == t))
            ends.append(np.arange(r[-1], r[0] - 1, -stride)[::-1])
        ends = np.concatenate(ends) if ends else np.zeros(0, np.int64)
        return ends[self.a["offset"][ends] >= self.L - 1]

    def _scored(self, ends):
        hit = np.zeros(len(self.score_rows), bool)
        for k in range(self.warmup, self.L):
            hit[ends - self.L + 1 + k] = True
        return hit & self.score_rows

    def _context_windows(self, ctx_rows, method):
        """(start, end) rows of every possible context whose last row is in ctx_rows."""
        if method == "cpd":
            key = f"regimes_{self.Lc}_end"
            if key not in self.a:
                raise ValueError(f"no CPD regimes with max length {self.Lc} in the arrays "
                                 "file: run `python -m project.features --steps cpd` first")
            s, e = self.a[f"regimes_{self.Lc}_start"], self.a[key]
            ok = ctx_rows[e]
            return s[ok], e[ok]
        e = np.flatnonzero(ctx_rows & (self.a["offset"] >= self.Lc - 1))
        return e - self.Lc + 1, e

    def __len__(self):
        return len(self.ends)

    def n_batches(self, batch_size):
        return -(-len(self.ends) // batch_size)

    def batches(self, batch_size, rng=None, shuffle=None):
        """Yield every target sequence once, `batch_size` at a time. Shuffled for train."""
        rng = rng if rng is not None else np.random.default_rng()
        shuffle = self.split == "train" if shuffle is None else shuffle
        order = rng.permutation(len(self.ends)) if shuffle else np.arange(len(self.ends))
        for i in range(0, len(order), batch_size):
            yield self.batch(order[i:i + batch_size], rng)

    def batch(self, idx, rng):
        """One mini-batch for target sequences `idx` (B targets, C contexts, Lc steps):

        x (B, L, 8)   y, next_ret, sigma, score_mask (B, L)   ticker_id (B,)   date (B, L)
        context_x (B, C, Lc, 8)   context_y, context_mask (B, C, Lc)   context_ticker_id (B, C)
        context_y is the contexts' vol-scaled next-day return (values use it, keys do not).
        Shorter contexts (CPD regimes) are padded with zeros at the front, mask False.
        """
        a, e = self.a, self.ends[idx]
        rows = e[:, None] - self.L + 1 + np.arange(self.L)
        k = _sample_distinct(self.n_avail[idx], self.size, rng)
        cs, ce = self.ctx_start[k], self.ctx_end[k]
        crow = ce[..., None] - self.Lc + 1 + np.arange(self.Lc)
        cmask = crow >= cs[..., None]
        crow = np.where(cmask, crow, ce[..., None])          # padding: any valid row, zeroed
        return {
            "x": a["inputs"][rows],
            "y": a["target"][rows],
            "next_ret": a["next_ret"][rows],
            "sigma": a["sigma"][rows],
            "score_mask": (np.arange(self.L) >= self.warmup) & self.score_rows[rows],
            "ticker_id": a["ticker_id"][e],
            "date": a["date"][rows],
            "context_x": a["inputs"][crow] * cmask[..., None],
            "context_y": a["target"][crow] * cmask,
            "context_mask": cmask,
            "context_ticker_id": a["ticker_id"][ce],
        }


def as_torch(batch, device="cpu"):
    """The numeric parts of a batch as torch tensors on `device` (dates are left out)."""
    import torch
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items() if k != "date"}


if __name__ == "__main__":
    import time
    config = features.load_config()
    model_cfg = load_model_config(data_cfg=config)
    arrays = load_arrays(config, model_cfg, rebuild=True)
    c = model_cfg["context"]
    print(f"context: {c['method']}, size {c['size']}, length {c.get('length')}, "
          f"anytime_in_training {c['anytime_in_training']}")
    for w in expanding_windows(model_cfg):
        parts = []
        for split in SPLITS:
            b = Batcher(arrays, w, split, model_cfg, config)
            parts.append(f"{split} {len(b):,} seqs (dropped {b.dropped:,}, "
                         f"unscored rows {b.unscored:,})")
        print(f"train from {w['train_start']}, test {w['test_start']} to {w['test_end']}: "
              + "; ".join(parts))
    b = Batcher(arrays, expanding_windows(model_cfg)[-1], "train", model_cfg, config)
    t0 = time.perf_counter()
    batch = next(b.batches(128, np.random.default_rng(0)))
    print(f"one batch of 128 in {time.perf_counter() - t0:.3f}s: "
          + ", ".join(f"{k} {v.shape}" for k, v in batch.items()))
