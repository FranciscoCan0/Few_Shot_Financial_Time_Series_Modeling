"""Benchmarks, portfolio returns and performance metrics (Wood et al., 2023: Section 2,
Eq. 2; benchmarks and metrics as in Section 4).

Everything works on a positions table with one row per (ticker, date):
    ticker, date, next_date, position, target, sigma
where `position` z_t is in [-1, 1], `target` is the vol-scaled next-day return
sigma_tgt / sigma_t * r_{t+1} and next_date is the day that return is earned.

    pos = benchmark_positions(arrays, rows)            # long, tsmom, macd for those rows
    daily = portfolio_returns(pos, "macd")             # one return per next_date
    metrics(daily)                                     # Sharpe, return, vol, drawdown
"""
import numpy as np
import pandas as pd

BENCHMARKS = ["long", "tsmom", "macd"]


def macd_response(y):
    """phi(y) = y exp(-y^2 / 4) / 0.89: strongest position at |y| = sqrt(2), fading for
    extreme signals (Baz et al., 2015, used by the paper's MACD benchmark)."""
    return y * np.exp(-y ** 2 / 4) / 0.89


def benchmark_positions(arrays, rows):
    """The three benchmark positions for the given rows of the arrays file:
      long   z = 1
      tsmom  z = sign of the past year's return (sign of ret_252)
      macd   z = average of phi(MACD signal) over the three (short, long) pairs
    Returns a positions table with columns long, tsmom, macd."""
    cols = list(arrays["input_columns"])
    x = arrays["inputs"][rows]
    macd = [i for i, c in enumerate(cols) if c.startswith("macd_")]
    return pd.DataFrame({
        "ticker": arrays["tickers"][arrays["ticker_id"][rows]],
        "date": arrays["date"][rows],
        "next_date": arrays["next_date"][rows],
        "target": arrays["target"][rows],
        "sigma": arrays["sigma"][rows],
        "long": 1.0,
        "tsmom": np.sign(x[:, cols.index("ret_252")]),
        "macd": macd_response(x[:, macd]).mean(axis=1),
    })


def portfolio_returns(pos, column="position", cost=0.0, vol_target=0.15):
    """Daily portfolio return, earned on next_date (Eq. 2): the equal-weighted average over
    the assets trading that day of z_t * sigma_tgt / sigma_t * r_{t+1}, minus the turnover
    cost C * sigma_tgt * |z_t / sigma_t - z_{t-1} / sigma_{t-1}| (sigma annualised) when
    cost > 0. The paper uses cost = 0."""
    p = pos.sort_values(["ticker", "date"])
    ret = p[column] * p["target"]
    if cost:
        lev = p[column] / (p["sigma"] * np.sqrt(252))
        turnover = lev.groupby(p["ticker"]).diff().abs().fillna(lev.abs())
        ret = ret - cost * vol_target * turnover
    return ret.groupby(p["next_date"]).mean().rename(column).sort_index()


def rescale(daily, vol_target=0.15):
    """Scale a daily return series to `vol_target` annual volatility over its whole history
    (the paper's reporting convention; ex post, so for display and comparison only)."""
    return daily * vol_target / (daily.std() * np.sqrt(252))


def drawdown(daily):
    """Drawdown of the cumulative (summed) returns from their running peak, <= 0."""
    wealth = daily.cumsum()
    return wealth - wealth.cummax()


def metrics(daily):
    """Annualised Sharpe ratio, return and volatility, and maximum drawdown."""
    ann_ret, ann_vol = daily.mean() * 252, daily.std() * np.sqrt(252)
    return pd.Series({"sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
                      "annual_return": ann_ret, "annual_vol": ann_vol,
                      "max_drawdown": drawdown(daily).min(), "days": len(daily)})


def compare(daily_by_strategy, periods=None, vol_target=0.15):
    """Metrics table for several strategies (dict name -> daily returns), each rescaled to
    `vol_target`, over the whole sample and optional periods {name: (start, end)}."""
    periods = {"all": (None, None), **(periods or {})}
    rows = {}
    for name, daily in daily_by_strategy.items():
        for period, (a, b) in periods.items():
            d = rescale(daily.loc[a:b], vol_target)
            rows[(name, period)] = metrics(d)
    return pd.DataFrame(rows).T.rename_axis(["strategy", "period"])


def forecast_scores(pred, quantiles=None):
    """Forecast quality on the scored test days: Gaussian negative log-likelihood (X-Trend-G,
    columns mu, sigma_forecast) and quantile (pinball) loss (X-Trend-Q, columns q_0 ...)."""
    out, y = {}, pred["target"].to_numpy()
    if {"mu", "sigma_forecast"} <= set(pred.columns):
        mu, s = pred["mu"].to_numpy(), pred["sigma_forecast"].to_numpy()
        out["gaussian_nll"] = np.mean(np.log(s) + 0.5 * ((y - mu) / s) ** 2
                                      + 0.5 * np.log(2 * np.pi))
        out["share_within_1_sigma"] = np.mean(np.abs(y - mu) <= s)   # ~0.68 if calibrated
    qcols = [c for c in pred.columns if c.startswith("q_")]
    if qcols and quantiles is not None:
        err = y[:, None] - pred[qcols].to_numpy()
        eta = np.asarray(quantiles)
        out["quantile_loss"] = np.maximum(eta * err, (eta - 1) * err).mean()
    return pd.Series(out, dtype=float)
