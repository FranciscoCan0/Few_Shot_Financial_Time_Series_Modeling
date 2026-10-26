"""Performance plots over time for daily strategy returns (from project.evaluation).

    daily = {"X-Trend": evaluation.portfolio_returns(pred), "TSMOM": ...}
    plot_performance(daily, path="results/replication/figures/run.png")
    plot_learning_curve({"train": [...], "val": [...]})

Every strategy is rescaled to the same annual volatility (15%) before plotting, as in the
paper, so the curves are comparable. Rescaling does not change Sharpe ratios.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project import evaluation as E


def rolling_sharpe(daily, window=252):
    """Annualised Sharpe ratio over the trailing `window` days (NaN until it is full)."""
    r = daily.rolling(window)
    return r.mean() / r.std() * np.sqrt(252)


def max_drawdown_to_date(daily):
    """The worst drawdown seen so far on each day (a running minimum of the drawdown)."""
    return E.drawdown(daily).cummin()


def rolling_max_drawdown(daily, window=252):
    """The worst drawdown within the trailing `window` days, measured from the peak inside
    that window."""
    wealth = daily.cumsum()
    return wealth.rolling(window).apply(lambda w: (w - np.maximum.accumulate(w)).min(),
                                        raw=True)


def plot_performance(daily_by_strategy, window=252, vol_target=0.15, main=None, title=None,
                     path=None):
    """Four panels over time for each strategy (dict name -> daily returns):
      1. cumulative return
      2. rolling `window`-day Sharpe ratio (and each strategy's full-period Sharpe in
         the legend)
      3. drawdown, with the main strategy's maximum drawdown to date dashed
      4. rolling `window`-day maximum drawdown
    `main` (default: the first strategy) is drawn thicker. Saves to `path` if given.
    Returns the figure.
    """
    main = main or next(iter(daily_by_strategy))
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.3, 1, 1]})
    for name, daily in daily_by_strategy.items():
        d = E.rescale(daily.dropna(), vol_target)
        m = E.metrics(d)
        style = {"lw": 2.0 if name == main else 1.0, "alpha": 1 if name == main else 0.8}
        label = f"{name}  (Sharpe {m['sharpe']:.2f}, max DD {m['max_drawdown']:.0%})"
        axes[0].plot(d.index, d.cumsum(), label=label, **style)
        axes[1].plot(d.index, rolling_sharpe(d, window), **style)
        line, = axes[2].plot(d.index, E.drawdown(d), **style)
        if name == main:
            axes[2].plot(d.index, max_drawdown_to_date(d), ls="--", color=line.get_color(),
                         lw=1, label=f"{name}: max drawdown to date")
        axes[3].plot(d.index, rolling_max_drawdown(d, window), **style)

    axes[0].set_ylabel(f"cumulative return\n({vol_target:.0%} vol)")
    axes[1].set_ylabel(f"rolling {window}-day\nSharpe")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[2].set_ylabel("drawdown")
    axes[3].set_ylabel(f"rolling {window}-day\nmax drawdown")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[2].legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title(title or f"{main}: performance over time")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def plot_learning_curve(history, title="Learning curve", path=None):
    """Loss per epoch for each series in `history` (dict name -> list of losses), e.g.
    {"train": [...], "val": [...]}. The best epoch of 'val' is marked. Returns the figure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, losses in history.items():
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, marker="o", ms=3, label=name)
        if name == "val" and len(losses):
            best = int(np.nanargmin(losses))
            ax.scatter(epochs[best], losses[best], s=80, facecolors="none", edgecolors="k",
                       zorder=3, label=f"best val (epoch {best + 1})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (lower is better)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    return fig


def performance_by_year(daily_by_strategy, vol_target=0.15):
    """Sharpe ratio per calendar year for each strategy (rows: years)."""
    out = {}
    for name, daily in daily_by_strategy.items():
        d = E.rescale(daily.dropna(), vol_target)
        out[name] = d.groupby(pd.DatetimeIndex(d.index).year).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252))
    return pd.DataFrame(out)
