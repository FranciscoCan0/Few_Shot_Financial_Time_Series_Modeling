"""Interactive dashboard for backtests: watch a running walk-forward or explore a finished one.

    streamlit run scripts/07_streamlit_dashboard.py

Reads results/replication/live/ (runs of 06_backtest_dashboard.py, refreshed while they
train) and results/replication/tables/ (finished 03_replicate.py runs). The sidebar picks
the run, the strategies, the date range, the rolling window, the volatility target and a
transaction cost; every number and chart below follows those choices.

Tabs:
  Performance     metrics table, cumulative return, rolling Sharpe, drawdown
  By period       Sharpe per test window and per calendar year
  Assets          Sharpe per asset, and one asset's positions and returns over time
  Training        validation loss and Sharpe per epoch for each window (live runs)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from project import evaluation as E
from project import features
from project import plotting as P
from project import vectorization_batching as V
from project.data import ROOT

LIVE = ROOT / "results" / "replication" / "live"
TABLES = ROOT / "results" / "replication" / "tables"

st.set_page_config(page_title="X-Trend backtest", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource
def arrays():
    cfg = features.load_config()
    return V.load_arrays(cfg, V.load_model_config(data_cfg=cfg))


@st.cache_data
def predictions(path, mtime):                        # mtime: reload when the file changes
    return pd.read_parquet(path)


@st.cache_data
def benchmarks(path, mtime):
    """Benchmark positions on the same rows as the run's predictions."""
    return E.benchmark_positions(arrays(), predictions(path, mtime)["row"].to_numpy())


def runs():
    """Run name -> its files, newest first."""
    out = {}
    for p in LIVE.glob("*_predictions.parquet"):
        out[p.name.removesuffix("_predictions.parquet")] = {"pred": p}
    for p in LIVE.glob("*_status.json"):
        out.setdefault(p.name.removesuffix("_status.json"), {})["status"] = p
    for p in TABLES.glob("*_predictions.parquet"):
        name = p.name.removesuffix("_predictions.parquet")
        status = out.get(name, {}).get("status")
        if status and json.loads(status.read_text())["state"] == "running":
            continue                                 # an 03_replicate.py --live run
        out[name] = {
            "pred": p, "log": TABLES / p.name.replace("_predictions.parquet",
                                                      "_training_log.csv")}
    for name, f in out.items():
        if "status" in f:
            f["log"] = LIVE / f"{name}.jsonl"
    newest = {n: max(p.stat().st_mtime for p in f.values() if p.exists())
              for n, f in out.items()}
    return dict(sorted(out.items(), key=lambda kv: -newest[kv[0]]))


def read_jsonl(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line])


def line_chart(df, title, yfmt=None, starts=(), zero=False):
    fig = px.line(df, title=title)
    fig.update_layout(height=330, margin={"l": 10, "r": 10, "t": 40, "b": 10}, legend_title=None,
                      xaxis_title=None, yaxis_title=None, hovermode="x unified")
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    for s in starts:
        fig.add_vline(x=s, line_width=1, line_dash="dot", line_color="gray")
    if zero:
        fig.add_hline(y=0, line_width=1, line_color="black")
    return fig


# ------------------------------------------------------------------ sidebar
all_runs = runs()
st.sidebar.title("Backtest dashboard")
if not all_runs:
    st.info("No runs yet. Start one in a terminal:\n\n"
            "`.venv/bin/python scripts/06_backtest_dashboard.py`\n\n"
            "This page shows it once the run has started.")
    st.stop()

name = st.sidebar.selectbox("Run", list(all_runs))
files = all_runs[name]
status = json.loads(files["status"].read_text()) if files.get("status") else None
running = bool(status) and status["state"] == "running"
refresh = st.sidebar.toggle("Auto refresh", value=running)
every = st.sidebar.slider("Refresh every (seconds)", 5, 120, 15, disabled=not refresh)

if status:
    done, total = len(status["windows_done"]), len(status["windows"])
    st.sidebar.progress(done / total, text=f"{done}/{total} windows done")
    st.sidebar.caption(f"**{status['state']}**: {status['status']}  \n"
                       f"{status['minutes']:.0f} min, updated {status['updated'][11:]}")

if not files.get("pred") or not files["pred"].exists():
    st.title(name)
    st.info("Training the first window. Performance appears when it finishes; "
            "the Training tab shows progress now.")
    pred = None
else:
    key = (str(files["pred"]), files["pred"].stat().st_mtime)
    pred = predictions(*key)

# ------------------------------------------------------------------ returns
if pred is not None:
    bench = benchmarks(*key)
    st.sidebar.subheader("Strategies")
    strategies = st.sidebar.multiselect("Show", ["model", *E.BENCHMARKS],
                                        default=["model", *E.BENCHMARKS])
    vol = st.sidebar.slider("Volatility target", 0.05, 0.30, 0.15, 0.01, format="%.2f")
    cost_bp = st.sidebar.slider("Transaction cost (bp)", 0.0, 10.0, 0.0, 0.5)
    window = st.sidebar.select_slider("Rolling window (days)", [63, 126, 252, 504], 252)

    cost = cost_bp / 1e4
    raw = {"model": E.portfolio_returns(pred, "position", cost, vol)}
    raw.update({b: E.portfolio_returns(bench, b, cost, vol) for b in E.BENCHMARKS})
    raw = pd.DataFrame(raw).sort_index()
    raw.index = pd.to_datetime(raw.index)

    lo, hi = raw.index.min().date(), raw.index.max().date()
    start, end = st.sidebar.slider("Dates", lo, hi, (lo, hi), format="YYYY-MM-DD")
    daily = raw.loc[str(start):str(end), strategies].dropna(how="all")
    scaled = daily.apply(lambda d: E.rescale(d.dropna(), vol))
    starts = [pd.Timestamp(s) for s in sorted(pred["test_start"].unique())[1:]
              if start <= pd.Timestamp(s).date() <= end]

st.title(name)

tab_perf, tab_period, tab_assets, tab_train = st.tabs(
    ["Performance", "By period", "Assets", "Training"])

# ------------------------------------------------------------------ performance
with tab_perf:
    if pred is None:
        st.write("Waiting for the first finished window.")
    elif not strategies:
        st.write("Pick at least one strategy in the sidebar.")
    else:
        m = scaled.apply(E.metrics).T
        cols = st.columns(4)
        if "model" in m.index:
            best_bench = m.drop("model")["sharpe"].max() if len(m) > 1 else np.nan
            cols[0].metric("Model Sharpe", f"{m.loc['model', 'sharpe']:.2f}",
                           None if np.isnan(best_bench) else
                           f"{m.loc['model', 'sharpe'] - best_bench:+.2f} vs best benchmark")
            cols[1].metric("Annual return", f"{m.loc['model', 'annual_return']:.1%}")
            cols[2].metric("Max drawdown", f"{m.loc['model', 'max_drawdown']:.1%}")
        cols[3].metric("Test days", f"{len(daily):,}")
        st.dataframe(m.style.format({"sharpe": "{:.2f}", "annual_return": "{:.1%}",
                                     "annual_vol": "{:.1%}", "max_drawdown": "{:.1%}",
                                     "days": "{:,.0f}"}), width="stretch")
        st.caption(f"All strategies rescaled to {vol:.0%} annual volatility over the "
                   "selected dates (ex post, for comparison). Dotted lines: a new window "
                   "(a newly trained model) starts.")
        st.plotly_chart(line_chart(scaled.cumsum(), "Cumulative return", ".0%", starts),
                        width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(line_chart(scaled.apply(P.rolling_sharpe, window=window),
                                   f"Rolling {window}-day Sharpe", None, starts, zero=True),
                        width="stretch")
        c2.plotly_chart(line_chart(scaled.apply(E.drawdown), "Drawdown", ".0%", starts),
                        width="stretch")

# ------------------------------------------------------------------ by period
with tab_period:
    if pred is not None and strategies:
        s_by_window = daily.groupby(
            pd.cut(daily.index, [pd.Timestamp(s) for s in sorted(pred["test_start"].unique())]
                   + [pd.Timestamp.max], right=False), observed=True
        ).apply(lambda d: d.mean() / d.std() * np.sqrt(252))
        s_by_window.index = [f"{iv.left.year}-{iv.left.year + 4}" for iv in s_by_window.index]
        s_by_window = s_by_window.dropna(how="all")
        fig = px.bar(s_by_window, barmode="group", title="Sharpe per test window")
        fig.update_layout(height=350, legend_title=None, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

        by_year = daily.groupby(daily.index.year).apply(
            lambda d: d.mean() / d.std() * np.sqrt(252))
        fig = px.imshow(by_year.T, text_auto=".1f", aspect="auto",
                        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        title="Sharpe per calendar year")
        fig.update_layout(height=120 + 45 * len(strategies), xaxis_title=None,
                          yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        if "model" in by_year:
            beat = (by_year["model"] > by_year.drop(columns="model").max(axis=1)).mean()
            st.caption(f"The model has the best Sharpe of the shown strategies in "
                       f"{beat:.0%} of years.")

# ------------------------------------------------------------------ assets
with tab_assets:
    if pred is not None:
        p = pred.join(bench[list(E.BENCHMARKS)])
        p["next_date"] = pd.to_datetime(p["next_date"])
        p = p[(p["next_date"].dt.date >= start) & (p["next_date"].dt.date <= end)]
        per_asset = pd.DataFrame({"model": p["position"] * p["target"],
                                  **{k: p[k] * p["target"] for k in E.BENCHMARKS}})
        g = per_asset.groupby(p["ticker"])
        table = (g.mean() / g.std() * np.sqrt(252)).assign(days=g.size())
        table["model - tsmom"] = table["model"] - table["tsmom"]
        st.subheader("Sharpe per asset")
        st.dataframe(table.sort_values("model", ascending=False).style.format(
            "{:.2f}", subset=["model", *E.BENCHMARKS, "model - tsmom"]),
            width="stretch", height=300)

        ticker = st.selectbox("Asset", sorted(p["ticker"].unique()))
        a = p[p["ticker"] == ticker].set_index("next_date").sort_index()
        pos = a[["position", "tsmom", "macd"]].rename(columns={"position": "model"})
        c1, c2 = st.columns(2)
        c1.plotly_chart(line_chart(pos, f"{ticker}: position (-1 short, +1 long)",
                                   zero=True), width="stretch")
        c2.plotly_chart(line_chart(pd.DataFrame({
            "asset (vol-scaled)": a["target"].cumsum(),
            "model on asset": (a["position"] * a["target"]).cumsum()}),
            f"{ticker}: cumulative return"), width="stretch")

# ------------------------------------------------------------------ training
with tab_train:
    log_path = files.get("log")
    if log_path is None or not log_path.exists():
        st.write("No training log for this run.")
    elif log_path.suffix == ".csv":                  # a finished 03_replicate.py run
        log = pd.read_csv(log_path)
        st.write("Every model trained by the hyperparameter search (kept = in the "
                 "ensemble).")
        fig = px.strip(log, x="test_start", y="val_loss", color="kept",
                       hover_data=list(log.columns), title="Validation loss per model")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(log, width="stretch")
    else:
        log = read_jsonl(log_path)
        if log.empty:
            st.write("No finished epoch yet.")
        else:
            who = log["model"] if "model" in log else "seed " + log["seed"].astype(str)
            log["run"] = "test " + log["test_start"].astype(str) + " " + who
            c1, c2 = st.columns(2)
            for col, ax, title in [("val_loss", c1, "Validation loss per epoch"),
                                   ("val_sharpe", c2, "Validation Sharpe per epoch")]:
                fig = px.line(log, x="epoch", y=col, color="run", markers=True, title=title)
                fig.update_layout(height=350, legend_title=None, yaxis_title=None)
                ax.plotly_chart(fig, width="stretch")
            pick = st.selectbox("Window", log["run"].unique()[::-1])
            one = log[log["run"] == pick].set_index("epoch")
            best = one["val_loss"].idxmin()
            fig = line_chart(one[["train_loss", "val_loss"]],
                             f"{pick}: train vs validation loss (best epoch {best})")
            fig.add_vline(x=best, line_dash="dash", line_color="black")
            fig.update_layout(hovermode="x")
            st.plotly_chart(fig, width="stretch")
            st.caption("A train loss that keeps falling while validation stays flat means "
                       "the model is overfitting; the kept model is the one at the best "
                       "validation epoch.")

if refresh:
    time.sleep(every)
    st.rerun()
