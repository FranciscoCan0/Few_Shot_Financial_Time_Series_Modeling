# Few-Shot Financial Time Series Modeling

Replication of [X-Trend (Wood et al., 2023)](https://arxiv.org/abs/2310.10500): a trend-following model for futures that
looks at a "context set" of past regimes from other assets before taking a position.
We use daily front-month prices for 50 futures from WRDS (Datastream), 1990-2023, and
compare against Long, TSMOM and MACD benchmarks.

Where things stand:

- Data download, feature building and change-point detection (CPD) work.
- X-Trend and the baseline model are implemented, with the Sharpe, Gaussian and quantile losses.
- The walk-forward backtest and both dashboards run.
- Extensions (`04_extend.py`) haven't been picked yet.

## Setup

Needs Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Put your WRDS username in `configs/data.json` (replace `"temp"`). The first download
asks for your password.

`data/` and `results/` are gitignored, so a fresh clone has to run steps 1 and 2.

## Running it

Run everything from the repo root. The scripts are numbered in the order you run them.

### 1. Download prices

```bash
python scripts/01_get_data.py
```

Pulls the futures from WRDS and stitches the contracts into back-adjusted series.
Writes `data/raw/futures_close.parquet`.

### 2. Build features, CPD regimes and training arrays

```bash
python scripts/02_build_dataset.py                 # everything, ~3 hours (CPD)
python scripts/02_build_dataset.py --skip-cpd      # reuse saved CPD, rebuild features + arrays
python scripts/02_build_dataset.py --regimes-only  # re-cut saved CPD scores, takes seconds
```

The CPD step fits a GP change-point model on every day for every asset, which is why
it's slow. Run it once, then use `--skip-cpd` or `--regimes-only` when you change other
settings. Writes the following to `data/processed/`:

- `xtrend_features.parquet` (Volatility Scaled Returns, MACD Features, & Target Returns)
- `cpd_scores.parquet`
- `cpd_regimes.parquet`
- `xtrend_arrays.npz` (Pre-batched vectorized version of xtrend_features)

### 3. Replicate the paper

```bash
python scripts/03_replicate.py --quick                     # a quick few minutes
python scripts/03_replicate.py --model xtrend --loss sharpe  # X-Trend
python scripts/03_replicate.py --model xtrend --loss gauss   # X-Trend-G
python scripts/03_replicate.py --model xtrend --loss qre     # X-Trend-Q
python scripts/03_replicate.py --model baseline              # no context set
```

Trains an ensemble on each expanding window (train from 1990, test 1995-2000,
2000-2005, ..., 2020-2023) and trades the test periods. The full setting is 100 models
per window, so it needs a GPU.

Use these flags to cut the run down:

- `--windows 3 4 5`
- `--seeds`
- `--search-iterations`
- `--max-epochs`
- `--max-steps-per-epoch`
- `--device cpu|cuda|mps`

Output goes to `results/replication/`:

- metrics and Sharpe by year in `tables/`
- return and drawdown plots in `figures/`
- model weights in `models/`

To watch the search in the Streamlit dashboard while it runs, add `--live` and start
`07` in a second terminal:

```bash
python scripts/03_replicate.py --model xtrend --loss sharpe --live    # terminal 1
streamlit run scripts/07_streamlit_dashboard.py                      # terminal 2
```

While it trains, the Training tab shows train/val loss and val Sharpe for every model in
the search, and Performance fills in as each window finishes. Once the run is done, the
dashboard switches to the final results, and the Training tab shows every model's
validation loss and which ones made the ensemble.

### 4. Extensions

`scripts/04_extend.py` is a placeholder. The POTENTIAL ideas in it are:

- extending the data to 2024-2026 (complete)
- adding transaction costs
- picking the futures by liquidity at the start of the sample

### 5. Watch one model train (Preliminary test run for model training)

```bash
python scripts/05_dashboard.py                       # X-Trend, test window 2005-2010
python scripts/05_dashboard.py --loss qre --window 4 --epochs 30
python scripts/05_dashboard.py --no-window           # no GUI, just updates a PNG
```

Opens a matplotlib window showing these while one model trains:

- batch loss
- train/val loss
- validation Sharpe
- test-period performance against the benchmarks

The test period is only for watching and isn't used to pick the model. Closing the
window stops training.

### 6-7. Walk-forward backtest with the Streamlit dashboard

```bash
python scripts/06_backtest_dashboard.py                  # terminal 1
streamlit run scripts/07_streamlit_dashboard.py          # terminal 2
```

`06` trains one setting (no hyperparameter search) on every window and writes its
progress to `results/replication/live/`. Some useful flags:

- `--windows 3 4 5`
- `--model baseline`
- `--seeds 2`
- `--epochs`

Ctrl+C stops it, and finished windows are kept.

`07` reads both the live runs and the finished `03` runs. The sidebar picks the run,
strategies, dates, vol target and transaction cost. It has four tabs:

- Performance
- By period
- Assets
- Training

## Running modules directly

The `src/project` modules also run on their own, which is handy for one step:

```bash
python -m project.data                        # same as step 1
python -m project.features                    # features + CPD
python -m project.features --steps regimes    # re-segment saved CPD scores only
python -m project.vectorization_batching      # rebuild arrays, print split sizes
```

## Layout

```
configs/data.json     WRDS login, date range, universe, feature and CPD settings
configs/model.json    sequence lengths, context set, architecture, training, search space, backtest
src/project/
  data.py                    WRDS download and contract stitching
  features.py                returns, MACD, vol scaling, GP change-point detection, regimes
  vectorization_batching.py  flat arrays, expanding windows, batches with context sets
  models.py                  X-Trend, baseline, losses
  training.py                training loop, random search, ensembling, predictions
  evaluation.py              benchmarks, portfolio returns, metrics
  plotting.py                performance plots
scripts/              the numbered steps above
notebooks/            exploration and checks on the data pipeline and arrays
```
