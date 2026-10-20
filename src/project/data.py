"""Daily front-month futures from WRDS (Thomson Reuters Datastream), stitched with
backwards ratio adjustment.

Settings live in configs/data.json. Run as a script to download and save the raw file:
    python -m project.data
  raw: data/raw/futures_close.parquet  (prices, within-contract returns, adj_close)
Features, CPD regimes and context sets are built from this file in project.features.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "data.json"

REQUIRED_KEYS = ["wrds_library", "start_date", "end_date", "roll_buffer_calendar_days",
                 "raw_path", "universe"]
SECTORS = {"CM", "EQ", "FI", "FX"}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def bad_config(errors):
    if errors:
        raise ValueError("Bad config:\n  " + "\n  ".join(errors))


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Raise ValueError listing every problem in the raw-data settings, before touching WRDS."""
    bad_config([f"missing key '{k}'" for k in REQUIRED_KEYS if k not in cfg])
    errors = []
    try:
        if pd.Timestamp(cfg["start_date"]) >= pd.Timestamp(cfg["end_date"]):
            errors.append("start_date must be before end_date")
    except ValueError as e:
        errors.append(f"bad date: {e}")

    buf = cfg["roll_buffer_calendar_days"]
    if not _is_int(buf) or buf < 0:
        errors.append(f"roll_buffer_calendar_days must be a non-negative integer, got {buf!r}")

    if Path(cfg["raw_path"]).is_absolute():
        errors.append("raw_path must be relative to the project root")

    seen = {}
    for ticker, a in cfg["universe"].items():
        code = a.get("contrcode")
        if not _is_int(code):
            errors.append(f"{ticker}: contrcode must be an integer, got {code!r}")
        elif code in seen:
            errors.append(f"{ticker}: contrcode {code} already used by {seen[code]}")
        else:
            seen[code] = ticker
        if a.get("sector") not in SECTORS:
            errors.append(f"{ticker}: sector must be one of {sorted(SECTORS)}, "
                          f"got {a.get('sector')!r}")
    bad_config(errors)


# --------------------------------------------------------------------------
# raw data
# --------------------------------------------------------------------------

def get_futures_close(cfg=None, db=None):
    """One row per (ticker, date) for the contract held that day (see _hold_front):
    the nearest-expiry contract at least `roll_buffer_calendar_days` before its last
    trading day, held until it gets that close, and never rolled backwards.

    Columns: date, ticker, contrcode, sector, futcode (contract held), close (its
    settlement price) and ret (its own return since its previous settlement).
    ret is computed within each contract before the held one is picked, so on a roll
    day it is the new contract's move, not the gap between two contracts.
    """
    cfg = cfg or load_config()
    lib, universe = cfg["wrds_library"], cfg["universe"]
    codes = ", ".join(str(a["contrcode"]) for a in universe.values())

    if db is None:
        import wrds  # only needed here, keeps the CPD worker processes light
        user = cfg.get("wrds_username")
        db = wrds.Connection(**({"wrds_username": user} if user else {}))

    # wrds_fut_contract stores each (futcode, date) several times: dedupe before LAG,
    # otherwise LAG compares a row with its own copy and returns 0
    df = db.raw_sql(f"""
        WITH info AS (
            SELECT DISTINCT futcode, contrcode, lasttrddate
            FROM {lib}.wrds_contract_info
            WHERE contrcode IN ({codes})
        ),
        f AS (
            SELECT DISTINCT ON (futcode, date_) futcode, date_, settlement
            FROM {lib}.wrds_fut_contract
            WHERE futcode IN (SELECT futcode FROM info)
              AND date_ BETWEEN '{cfg["start_date"]}' AND '{cfg["end_date"]}'
              AND settlement IS NOT NULL
            ORDER BY futcode, date_
        ),
        px AS (
            SELECT i.contrcode, i.lasttrddate, f.futcode, f.date_, f.settlement,
                   f.settlement / NULLIF(LAG(f.settlement) OVER (
                       PARTITION BY f.futcode ORDER BY f.date_), 0) - 1 AS ret
            FROM f JOIN info i ON f.futcode = i.futcode
        )
        SELECT contrcode, date_ AS date, futcode, settlement AS close, ret, lasttrddate
        FROM px
        WHERE lasttrddate >= date_ + {int(cfg["roll_buffer_calendar_days"])}
    """, date_cols=["date", "lasttrddate"])

    # a mistyped contrcode matches nothing in WRDS, so it comes back with no rows
    df["contrcode"] = df["contrcode"].astype(int)
    empty = [f"{t} ({a['contrcode']})" for t, a in universe.items()
             if a["contrcode"] not in set(df["contrcode"])]
    if empty:
        raise ValueError(f"No prices in {cfg['start_date']}..{cfg['end_date']} for: "
                         f"{', '.join(empty)}. Check these contrcodes in configs/data.json.")

    by_code = {a["contrcode"]: (t, a["sector"]) for t, a in universe.items()}
    df["ticker"] = df["contrcode"].map(lambda c: by_code[c][0])
    df["sector"] = df["contrcode"].map(lambda c: by_code[c][1])
    df["futcode"] = df["futcode"].astype(int)
    df["close"] = df["close"].astype(float)
    df["ret"] = df["ret"].astype(float)
    df = _hold_front(df, int(cfg["roll_buffer_calendar_days"]))
    return (df[["date", "ticker", "contrcode", "sector", "futcode", "close", "ret"]]
            .sort_values(["ticker", "date"]).reset_index(drop=True))


def _hold_front(df, buffer_days):
    """Pick one contract per (ticker, date) from all eligible contracts that have a price.

    Hold a contract until it is within `buffer_days` of its last trading day, then roll
    to the nearest contract expiring after it. Never roll backwards: a nearer contract
    that only starts having prices later is ignored. On a day the held contract has no
    price, the day is skipped rather than filled from another contract, which would
    create two fake rolls and count the move between them twice.
    """
    buffer = np.timedelta64(buffer_days, "D")
    df = df.sort_values(["ticker", "date", "lasttrddate", "futcode"]).reset_index(drop=True)
    keep = []
    for _, g in df.groupby("ticker", sort=False):
        idx, dates = g.index.to_numpy(), g["date"].to_numpy()
        exp, fut = g["lasttrddate"].to_numpy(), g["futcode"].to_numpy()
        starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
        held, held_exp = None, None
        for a, b in zip(starts, np.r_[starts[1:], len(g)]):   # one block per date
            if held is not None and held_exp >= dates[a] + buffer:   # keep holding
                hit = np.flatnonzero(fut[a:b] == held)
                if len(hit):
                    keep.append(idx[a + hit[0]])
                continue                                             # no price: skip day
            later = np.arange(a, b) if held is None else a + np.flatnonzero(exp[a:b] > held_exp)
            if len(later):                                           # roll (or first day)
                held, held_exp = fut[later[0]], exp[later[0]]
                keep.append(idx[later[0]])
    return df.loc[keep]


def back_adjust(df):
    """Add adj_close: backwards ratio-adjusted price (the paper's stitching method).

    At each roll, ratio = new contract's price / old contract's price on the day before
    the switch. Every price before the roll is multiplied by that ratio, so each day's
    factor is the product of the ratios of all later rolls. The latest adj_close equals
    the real price, and every adjusted daily return equals the held contract's return.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    ret = df["ret"].fillna(0.0)                     # first day of a contract's history
    g = df.groupby("ticker")
    is_roll = df["futcode"] != g["futcode"].shift()
    prev_close = g["close"].shift()                 # old contract, day before the roll
    new_prev = df["close"] / (1 + ret)              # new contract, day before the roll
    ratio = pd.Series(np.where(is_roll & prev_close.notna(), new_prev / prev_close, 1.0),
                      index=df.index)
    factor = ratio.groupby(df["ticker"]).transform(
        lambda s: s[::-1].cumprod()[::-1].shift(-1, fill_value=1.0))
    df["adj_close"] = df["close"] * factor
    return df


def make_raw(cfg=None, db=None):
    """Download, back-adjust and save the raw data. Returns the DataFrame."""
    cfg = cfg or load_config()
    df = back_adjust(get_futures_close(cfg, db))
    out = ROOT / cfg["raw_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"saved {out.relative_to(ROOT)}: {len(df):,} rows, {df.ticker.nunique()} tickers")
    return df


if __name__ == "__main__":
    make_raw(load_config())
