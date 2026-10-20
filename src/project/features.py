"""X-Trend inputs (Wood et al., 2023), GP change-point regimes and context sets, built
from the raw file that project.data saves.

Settings live in configs/data.json. Run as a script to build the processed files:
    python -m project.features                   # processed + cpd
    python -m project.features --steps regimes   # re-segment saved CPD scores only
  processed: data/processed/xtrend_features.parquet  (returns, MACD, targets)
             data/processed/cpd_scores.parquet       (GP change-point score per day, slow)
             data/processed/cpd_regimes.parquet      (regimes from Algorithm 1)
Context sets (Section 3.2) are sampled at training time with ContextSampler.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit

from project import data
from project.data import ROOT, _is_int, bad_config

REQUIRED_KEYS = ["processed_path", "cpd_scores_path", "regimes_path",
                 "features", "cpd", "contexts"]
REQUIRED_FEATURE_KEYS = ["sample_start", "vol_span", "vol_target", "return_horizons",
                         "macd_pairs", "macd_price_std_window", "macd_signal_std_window",
                         "winsorize", "winsorize_halflife", "winsorize_sds"]
REQUIRED_CPD_KEYS = ["lookback", "min_length", "location_starts", "segmentations", "n_jobs"]
REQUIRED_CONTEXT_KEYS = ["target_length", "sizes", "final_hidden_lengths", "cpd_max_lengths"]
PATH_KEYS = ["processed_path", "cpd_scores_path", "regimes_path"]
# target: vol-scaled next-day return (the model's y); next_ret: unscaled r_{t+1};
# sigma: ex-ante daily vol at t. None of these are model inputs.
TARGET_COLUMNS = ["target", "next_ret", "sigma"]


def load_config(path=data.CONFIG_PATH):
    """The raw-data settings checked by project.data, plus the feature settings."""
    cfg = data.load_config(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Raise ValueError listing every problem in the feature, CPD and context settings."""
    errors = [f"missing key '{k}'" for k in REQUIRED_KEYS if k not in cfg]
    for block, keys in [("features", REQUIRED_FEATURE_KEYS), ("cpd", REQUIRED_CPD_KEYS),
                        ("contexts", REQUIRED_CONTEXT_KEYS)]:
        errors += [f"missing key '{block}.{k}'" for k in keys if k not in cfg.get(block, {})]
    bad_config(errors)

    try:
        sample_start = pd.Timestamp(cfg["features"]["sample_start"])
        if not pd.Timestamp(cfg["start_date"]) <= sample_start < pd.Timestamp(cfg["end_date"]):
            errors.append("features.sample_start must be between start_date and end_date")
    except ValueError as e:
        errors.append(f"bad date: {e}")

    for key in PATH_KEYS:
        if Path(cfg[key]).is_absolute():
            errors.append(f"{key} must be relative to the project root")

    f = cfg["features"]
    for key in ["vol_span", "macd_price_std_window", "macd_signal_std_window",
                "winsorize_halflife"]:
        if not _is_int(f[key]) or f[key] < 2:
            errors.append(f"features.{key} must be an integer >= 2, got {f[key]!r}")
    for key in ["vol_target", "winsorize_sds"]:
        if not isinstance(f[key], (int, float)) or isinstance(f[key], bool) or f[key] <= 0:
            errors.append(f"features.{key} must be a positive number, got {f[key]!r}")
    if not isinstance(f["winsorize"], bool):
        errors.append(f"features.winsorize must be true or false, got {f['winsorize']!r}")
    h = f["return_horizons"]
    if not isinstance(h, list) or not h or not all(_is_int(x) and x >= 1 for x in h):
        errors.append(f"features.return_horizons must be a list of positive integers, got {h!r}")
    pairs = f["macd_pairs"]
    if not isinstance(pairs, list) or not pairs or not all(
            isinstance(p, list) and len(p) == 2 and all(_is_int(x) and x >= 2 for x in p)
            and p[0] < p[1] for p in pairs):
        errors.append("features.macd_pairs must be a list of [short, long] integer pairs "
                      f"with 2 <= short < long, got {pairs!r}")

    c = cfg["cpd"]
    for key, lo in [("lookback", 5), ("min_length", 1), ("location_starts", 1), ("n_jobs", 1)]:
        if not _is_int(c[key]) or c[key] < lo:
            errors.append(f"cpd.{key} must be an integer >= {lo}, got {c[key]!r}")
    segs = c["segmentations"]
    if not isinstance(segs, list) or not segs or not all(
            isinstance(s, dict) and set(s) == {"threshold", "max_length"}
            and isinstance(s["threshold"], (int, float)) and 0.5 <= s["threshold"] < 1
            and _is_int(s["max_length"]) and s["max_length"] > c["min_length"] for s in segs):
        errors.append("cpd.segmentations must be a list of {threshold, max_length} with "
                      f"0.5 <= threshold < 1 and max_length > cpd.min_length, got {segs!r}")

    x = cfg["contexts"]
    for key in ["sizes", "final_hidden_lengths", "cpd_max_lengths"]:
        if not isinstance(x[key], list) or not x[key] or not all(
                _is_int(v) and v >= 1 for v in x[key]):
            errors.append(f"contexts.{key} must be a list of positive integers, got {x[key]!r}")
    if not _is_int(x["target_length"]) or x["target_length"] < 1:
        errors.append(f"contexts.target_length must be a positive integer, "
                      f"got {x['target_length']!r}")
    if isinstance(segs, list) and isinstance(x["cpd_max_lengths"], list):
        have = {s.get("max_length") for s in segs if isinstance(s, dict)}
        missing = [m for m in x["cpd_max_lengths"] if m not in have]
        if missing:
            errors.append(f"contexts.cpd_max_lengths {missing} have no matching "
                          "cpd.segmentations entry")
    bad_config(errors)


# --------------------------------------------------------------------------
# processed data: X-Trend inputs and target
# --------------------------------------------------------------------------

def feature_columns(cfg):
    f = cfg["features"]
    return ([f"ret_{h}" for h in f["return_horizons"]]
            + [f"macd_{s}_{l}" for s, l in f["macd_pairs"]])


def _halflife(timescale):
    # Section 2.2: half-life = log(1/2) / log(1 - 1/S)
    return np.log(0.5) / np.log(1 - 1 / timescale)


def _ticker_features(p, f):
    """Inputs and target for one ticker from its adjusted price series p (date index)."""
    if f["winsorize"]:   # clip prices more than N EWM std from their EWM mean
        ew = p.ewm(halflife=f["winsorize_halflife"])
        band = f["winsorize_sds"] * ew.std()
        p = p.clip(ew.mean() - band, ew.mean() + band)

    r = p.pct_change()                                                   # Eq. 1
    sigma = r.ewm(span=f["vol_span"], min_periods=f["vol_span"]).std()   # ex-ante daily vol

    out = pd.DataFrame(index=p.index)
    for h in f["return_horizons"]:                                       # Eq. 5
        out[f"ret_{h}"] = p.pct_change(h) / (sigma * np.sqrt(h))
    for s, l in f["macd_pairs"]:                                         # Eqs. 4a-4b
        m = ((p.ewm(halflife=_halflife(s)).mean() - p.ewm(halflife=_halflife(l)).mean())
             / p.rolling(f["macd_price_std_window"]).std())
        out[f"macd_{s}_{l}"] = m / m.rolling(f["macd_signal_std_window"]).std()

    # targets. Eqs. 8, 20 and the quantile loss all use the vol-scaled next-day return
    # sigma_tgt / sigma_t * r_{t+1} (sigma annualised so the units match vol_target)
    out["next_ret"] = r.shift(-1)                                        # r_{t+1}
    out["target"] = out["next_ret"] * f["vol_target"] / (sigma * np.sqrt(252))
    out["sigma"] = sigma          # ex-ante daily vol: leverage and turnover cost in Eq. 2
    return out


def build_features(raw, cfg=None):
    """X-Trend inputs (vol-scaled returns, MACD), ticker_id for the ticker embedding, and
    the targets (vol-scaled and raw next-day return, sigma), one row per (ticker, date).

    Every input only uses data up to its own date, so the table can be built once and
    batched from directly: no recomputation per roll or per training step.
    """
    cfg = cfg or load_config()
    f = cfg["features"]
    missing = {"date", "ticker", "sector", "adj_close"} - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns {sorted(missing)}; run make_raw first")

    cols = feature_columns(cfg)
    panel = (raw.sort_values(["ticker", "date"])
                .groupby("ticker")[["date", "adj_close"]]
                .apply(lambda g: _ticker_features(g.set_index("date")["adj_close"], f))
                .reset_index())
    panel["sector"] = panel["ticker"].map(raw.groupby("ticker")["sector"].first())
    panel["ticker_id"] = panel["ticker"].astype("category").cat.codes

    # drop the warm-up rows and the last day (no next-day return)
    panel = (panel.dropna(subset=cols + ["target"])
                  .query("@f['sample_start'] <= date <= @cfg['end_date']")
                  .reset_index(drop=True))
    return panel[["date", "ticker", "ticker_id", "sector", *cols, *TARGET_COLUMNS]]


def make_processed(cfg=None, raw=None):
    """Build and save the model inputs from the raw file (or a given raw DataFrame)."""
    cfg = cfg or load_config()
    if raw is None:
        raw = pd.read_parquet(ROOT / cfg["raw_path"])
    panel = build_features(raw, cfg)
    out = ROOT / cfg["processed_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"saved {out.relative_to(ROOT)}: {len(panel):,} rows, {panel.ticker.nunique()} "
          f"tickers, {panel.date.min().date()} to {panel.date.max().date()}")
    return panel


# --------------------------------------------------------------------------
# change-point detection: GP scores (Appendix A) and regimes (Algorithm 1)
# --------------------------------------------------------------------------

# hyperparameters are fitted in log space, inside bounds that suit standardised returns
_LOG2PI = np.log(2 * np.pi)
_B_LOGVAR, _B_LOGLS, _B_LOGNOISE = (-6, 3), (np.log(0.1), np.log(1e3)), (-9, 2)
_B_LOGSTEEP = (np.log(0.1), np.log(100))


def _matern32(D, log_var, log_ls):
    """Matérn 3/2 kernel on distances D, and its derivative w.r.t. log lengthscale."""
    a = np.sqrt(3) * D / np.exp(log_ls)
    e = np.exp(log_var - a)
    return e * (1 + a), e * a * a


def _nll_and_grad(K, y, dKs):
    """-log N(y | 0, K) and its gradient, given dK/dθ for each hyperparameter θ."""
    try:
        c = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros(len(dKs))
    alpha = cho_solve(c, y, check_finite=False)
    W = cho_solve(c, np.eye(len(y)), check_finite=False) - np.outer(alpha, alpha)
    nll = 0.5 * y @ alpha + np.log(np.diag(c[0])).sum() + 0.5 * len(y) * _LOG2PI
    return nll, np.array([0.5 * np.sum(W * dK) for dK in dKs])


def _matern_nll(th, y, D, I):
    """One regime. th = log [variance, lengthscale, noise]."""
    k, dk_ls = _matern32(D, th[0], th[1])
    noise = np.exp(th[2])
    return _nll_and_grad(k + noise * I, y, [k, dk_ls, noise * I])


def _changepoint_nll(th, y, D, I, x):
    """Two regimes: GP1 before location c, GP2 after, with a sigmoid switch between them.
    th = [log var1, log ls1, log var2, log ls2, c, log steepness, log noise]."""
    k1, dk1 = _matern32(D, th[0], th[1])
    k2, dk2 = _matern32(D, th[2], th[3])
    c, steep, noise = th[4], np.exp(th[5]), np.exp(th[6])
    s = expit(steep * (x - c))      # weight on GP2: 0 before the change, 1 after
    q = 1 - s
    A, B = np.outer(q, q), np.outer(s, s)

    def dK(ds):                     # dK for a parameter that moves the switch, ds = ds/dθ
        return (-(np.outer(ds, q) + np.outer(q, ds)) * k1
                + (np.outer(ds, s) + np.outer(s, ds)) * k2)

    sq = s * q
    return _nll_and_grad(A * k1 + B * k2 + noise * I, y,
                         [A * k1, A * dk1, B * k2, B * dk2,
                          dK(-steep * sq), dK(steep * (x - c) * sq), noise * I])


def _fit(f, th0, args, bounds):
    return minimize(f, th0, args, jac=True, method="L-BFGS-B", bounds=bounds)


def cpd_window(returns, location_starts=10):
    """Change-point score for one lookback window of daily returns.

    Fits a Matérn 3/2 GP (one regime, likelihood L_M) and a change-point GP (two
    regimes, L_C) by maximum likelihood, as in Wood et al. (2021) which X-Trend follows.
    Returns (severity, location): severity = L_C / (L_M + L_C), location = position of
    the change in days from the window start. The change-point fit is restarted from
    `location_starts` evenly spaced locations because it has several local optima.
    """
    r = np.asarray(returns, dtype=float)
    if not np.isfinite(r).all() or r.std() == 0:
        return 0.0, np.nan          # flat window (stale prices): nothing to detect
    y = (r - r.mean()) / r.std()
    n = len(y)
    x = np.arange(n, dtype=float)
    D, I = np.abs(x[:, None] - x[None, :]), np.eye(n)

    m = min((_fit(_matern_nll, th0, (y, D, I), [_B_LOGVAR, _B_LOGLS, _B_LOGNOISE])
             for th0 in (np.log([1.0, 1.0, 0.1]), np.log([0.5, 5.0, 0.5]))),
            key=lambda res: res.fun)
    lv, ll, ln = m.x                # both sides of the change start from the one-regime fit
    bounds = [_B_LOGVAR, _B_LOGLS, _B_LOGVAR, _B_LOGLS, (0, n - 1), _B_LOGSTEEP, _B_LOGNOISE]
    cp = min((_fit(_changepoint_nll, [lv, ll, lv, ll, c0, 0.0, ln], (y, D, I, x), bounds)
              for c0 in np.linspace(0, n - 1, location_starts + 2)[1:-1]),
             key=lambda res: res.fun)
    # L_C / (L_M + L_C) = 1 / (1 + exp(log L_M - log L_C)), and .fun = -log L
    return float(expit(m.fun - cp.fun)), float(cp.x[4])


def _cpd_chunk(task):
    ticker, dates, windows, location_starts = task
    return ticker, dates, np.array([cpd_window(w, location_starts) for w in windows])


def cpd_scores(raw, cfg=None):
    """Severity and change-point location for every (ticker, day) from sample_start on.

    Day t uses the `lookback` daily returns ending at t (x_{-lbw:t} in the paper), so there
    is no look-ahead. cp_days_back = how many days before t the change sits. This is the
    slow part (two GP fits per day), so it runs once in parallel and is saved; Algorithm 1
    then turns it into regimes for any threshold or max length in seconds.
    """
    cfg = cfg or load_config()
    c = cfg["cpd"]
    lbw = c["lookback"]
    sample_start = np.datetime64(pd.Timestamp(cfg["features"]["sample_start"]))

    tasks = []
    for ticker, g in raw.sort_values(["ticker", "date"]).groupby("ticker"):
        r = g["adj_close"].pct_change().to_numpy()      # r[0] is NaN, so t starts at lbw
        dates = g["date"].to_numpy()
        ends = [t for t in range(lbw, len(g)) if dates[t] >= sample_start]
        for i in range(0, len(ends), 500):
            e = ends[i:i + 500]
            tasks.append((ticker, dates[e], np.stack([r[t - lbw + 1:t + 1] for t in e]),
                          c["location_starts"]))

    # the matrices are tiny, so one BLAS thread per worker process
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(var, "1")
    parts, n_windows = [], sum(len(t[1]) for t in tasks)
    print(f"CPD: {n_windows:,} windows, {len(tasks)} chunks, {c['n_jobs']} workers", flush=True)
    with ProcessPoolExecutor(max_workers=c["n_jobs"]) as pool:
        for i, fut in enumerate(as_completed([pool.submit(_cpd_chunk, t) for t in tasks]), 1):
            ticker, dates, res = fut.result()
            parts.append(pd.DataFrame({"date": dates, "ticker": ticker, "severity": res[:, 0],
                                       "cp_days_back": (lbw - 1) - res[:, 1]}))
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} chunks done", flush=True)
    return pd.concat(parts).sort_values(["ticker", "date"]).reset_index(drop=True)


def segment_regimes(severity, cp_days_back, threshold, min_length, max_length):
    """Algorithm 1 (Appendix A) on one ticker's scores in date order.

    Walks back from the last day. A score >= threshold ends the current regime at the
    change point; otherwise the regime grows until it reaches max_length. Regimes shorter
    than min_length are dropped. Returns (start, end) row positions, both inclusive.
    """
    t = t1 = len(severity) - 1
    regimes = []
    while t >= 0:
        if severity[t] >= threshold:
            tc = t - cp_days_back[t]
            t0 = max(int(np.ceil(tc)), 0)
            if t1 - t0 >= min_length:
                regimes.append((t0, t1))
            t = int(np.floor(tc)) - 1   # skip the change so the next regime is clean
            t1 = t
        else:
            t -= 1
            if t1 - t > max_length:
                t = t1 - max_length
            if t1 - t == max_length and t >= 0:
                regimes.append((t, t1))
                t1 = t
    return regimes[::-1]


def cpd_regimes(scores, cfg=None):
    """Regimes for every ticker and every cpd.segmentations entry (threshold, max_length)."""
    cfg = cfg or load_config()
    c = cfg["cpd"]
    rows = []
    for seg in c["segmentations"]:
        for ticker, g in scores.sort_values(["ticker", "date"]).groupby("ticker"):
            dates = g["date"].to_numpy()
            for t0, t1 in segment_regimes(g["severity"].to_numpy(), g["cp_days_back"].to_numpy(),
                                          seg["threshold"], c["min_length"], seg["max_length"]):
                rows.append((ticker, seg["max_length"], seg["threshold"],
                             dates[t0], dates[t1], t1 - t0 + 1))
    cols = ["ticker", "max_length", "threshold", "start_date", "end_date", "length"]
    return (pd.DataFrame(rows, columns=cols)
              .sort_values(["max_length", "ticker", "start_date"]).reset_index(drop=True))


def make_cpd(cfg=None, raw=None, recompute_scores=True):
    """Compute (or reload) the CPD scores, segment them into regimes, save both."""
    cfg = cfg or load_config()
    scores_path, regimes_path = ROOT / cfg["cpd_scores_path"], ROOT / cfg["regimes_path"]
    if recompute_scores:
        if raw is None:
            raw = pd.read_parquet(ROOT / cfg["raw_path"])
        scores = cpd_scores(raw, cfg)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(scores_path)
        print(f"saved {scores_path.relative_to(ROOT)}: {len(scores):,} rows")
    else:
        if not scores_path.exists():
            raise FileNotFoundError(f"{scores_path.relative_to(ROOT)} not found: "
                                    "run `python -m project.features --steps cpd` first")
        scores = pd.read_parquet(scores_path)

    regimes = cpd_regimes(scores, cfg)
    regimes_path.parent.mkdir(parents=True, exist_ok=True)
    regimes.to_parquet(regimes_path)
    print(f"saved {regimes_path.relative_to(ROOT)}: {len(regimes):,} regimes")
    for (L, thr), g in regimes.groupby(["max_length", "threshold"]):
        used = g["length"].clip(upper=L)     # the sampler keeps at most max_length days
        print(f"  max_length {L}, threshold {thr}: {len(g):,} regimes, mean length used "
              f"{used.mean():.1f} (paper calibrates the threshold so this is about {L / 2:.0f})")
    return scores, regimes


# --------------------------------------------------------------------------
# context sets (Section 3.2)
# --------------------------------------------------------------------------

class ContextSampler:
    """Draws X-Trend context sets from the processed features (and CPD regimes).

    Each context is a sequence of the inputs plus `target` (the paper also feeds the
    contexts' next-day vol-scaled returns) and its ticker_id. The three methods:
      final_hidden(size, length)  F: random sequences of a fixed length (paper: 21 or 63);
                                     the model attends to each one's final hidden state.
      time_equivalent(size)       T: random sequences as long as the target (126); target
                                     step k attends to step k of every context.
      cpd(size, max_length)       C: random CPD regimes of at most 21 or 63 days; the model
                                     attends to each one's final hidden state.
    F and T only differ here in length; the difference in attention is in the model.

    Causality: at test time pass `before` = the target's last date, so every context ends
    strictly before it and its next-day return is already known. The paper does not
    enforce this in training; there, build the sampler on training-period rows only.
    For zero-shot, build it on the context assets only.

    Each method returns a dict: x (size, L, n_inputs + 1) float32, mask (size, L) True on
    real steps, ticker_id, ticker, start, end. Shorter sequences are padded at the front,
    so the last real step is always at position L - 1.
    """

    def __init__(self, panel, regimes=None, cfg=None):
        self.cfg = cfg or load_config()
        self.columns = feature_columns(self.cfg) + ["target"]
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
        self.values = panel[self.columns].to_numpy(np.float32)
        self.dates = panel["date"].to_numpy()
        self.tickers = panel["ticker"].to_numpy()
        self.ticker_id = panel["ticker_id"].to_numpy()
        # row offset within its ticker: a window of length L can end at rows with offset >= L-1
        self.offset = panel.groupby("ticker").cumcount().to_numpy()
        self._ends = {}

        self.regimes = {}
        if regimes is not None:
            pos = pd.Series(np.arange(len(panel)),
                            index=pd.MultiIndex.from_frame(panel[["ticker", "date"]]))
            for L, g in regimes.groupby("max_length"):
                s = pos.reindex(pd.MultiIndex.from_arrays([g["ticker"], g["start_date"]]))
                e = pos.reindex(pd.MultiIndex.from_arrays([g["ticker"], g["end_date"]]))
                ok = s.notna().to_numpy() & e.notna().to_numpy()   # inside the feature sample
                s, e = s.to_numpy()[ok].astype(int), e.to_numpy()[ok].astype(int)
                # Algorithm 1 can return max_length + 1 days: keep the most recent max_length
                self.regimes[int(L)] = (np.maximum(s, e - L + 1), e)

    def _gather(self, starts, ends, length):
        x = np.zeros((len(ends), length, len(self.columns)), np.float32)
        mask = np.zeros((len(ends), length), bool)
        for i, (s, e) in enumerate(zip(starts, ends)):
            k = e - s + 1
            x[i, length - k:] = self.values[s:e + 1]
            mask[i, length - k:] = True
        return {"x": x, "mask": mask, "ticker_id": self.ticker_id[ends],
                "ticker": self.tickers[ends], "start": self.dates[starts], "end": self.dates[ends]}

    def _pick(self, candidates, ends_at, size, before, rng):
        if before is not None:
            candidates = candidates[ends_at < pd.Timestamp(before).to_datetime64()]
        if len(candidates) < size:
            when = f" ending before {before}" if before is not None else ""
            raise ValueError(f"only {len(candidates)} contexts available{when}, need {size}")
        return (rng or np.random.default_rng()).choice(candidates, size, replace=False)

    def _random_windows(self, size, length, before, rng):
        if length not in self._ends:
            self._ends[length] = np.flatnonzero(self.offset >= length - 1)
        ends = self._ends[length]
        e = self._pick(ends, self.dates[ends], size, before, rng)
        return self._gather(e - length + 1, e, length)

    def final_hidden(self, size, length, before=None, rng=None):
        """F: `size` random sequences of `length` days from any asset."""
        return self._random_windows(size, length, before, rng)

    def time_equivalent(self, size, before=None, rng=None):
        """T: `size` random sequences as long as the target sequence."""
        return self._random_windows(size, self.cfg["contexts"]["target_length"], before, rng)

    def cpd(self, size, max_length, before=None, rng=None):
        """C: `size` random CPD regimes from the segmentation with this max_length."""
        if max_length not in self.regimes:
            raise ValueError(f"no regimes with max_length {max_length}; have "
                             f"{sorted(self.regimes)} (pass regimes= when building the sampler)")
        s, e = self.regimes[max_length]
        i = self._pick(np.arange(len(e)), self.dates[e], size, before, rng)
        return self._gather(s[i], e[i], max_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the X-Trend processed files.")
    parser.add_argument("--steps", nargs="+", default=["processed", "cpd"],
                        choices=["processed", "cpd", "regimes"],
                        help="processed: features; cpd: GP scores + regimes (slow); "
                             "regimes: re-segment saved scores only")
    args = parser.parse_args()
    config = load_config()
    if "processed" in args.steps:
        make_processed(config)
    if "cpd" in args.steps:
        make_cpd(config)
    elif "regimes" in args.steps:
        make_cpd(config, recompute_scores=False)
