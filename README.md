# Few-Shot Financial Time Series Modeling

Replication of [X-Trend (Wood et al., 2023)](https://arxiv.org/abs/2310.10500): a trend-following model for futures that
looks at a "context set" of past regimes from other assets before taking a position.
We use daily front-month prices for 50 futures from WRDS (Datastream), 1990-2025 (49 have
enough history to use), backtest 1995-2023, and compare against Long, TSMOM and MACD benchmarks.

Team Members: Cole Pridgen, Francisco Canova, Noah Apteker

- **Note for this submission:** to finish training before the deadline, we ran a reduced version of the paper's setup (ensembles of 2 models with 1 hyperparameter draw each and at most 300 batches per epoch, vs. the paper's 10 x 10 search with no batch maximum). The full search for all 48 variants would take weeks on our single GPU. See [Preliminary Replication Results](#preliminary-replication-results) for the exact settings.

Where things stand:

- Data download, feature building and change-point detection (CPD) work.
- X-Trend and the baseline model are implemented, with the Sharpe, Gaussian and quantile losses.
- The walk-forward backtest and both dashboards run.
- Preliminary results for the four main models are in [Preliminary Replication Results](#preliminary-replication-results)
  (reduced training budget; full-budget runs pending).
- Extensions (`04_extend.py`) haven't been picked yet.

## Setup

Needs Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

On Windows with an NVIDIA GPU, install the CUDA build of PyTorch first
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`), otherwise
training runs on the CPU.

Put your WRDS username in `configs/data.json` (replace `"temp"`). The first download
asks for your password.

`data/` is gitignored, so a fresh clone has to run steps 1 and 2. The result tables and
figures of the preliminary runs are committed in `results/replication/`; the large outputs
(predictions, model weights, live progress files) are not.

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
settings. `cpd.n_jobs` in `configs/data.json` sets how many CPU cores it uses. Writes the
following to `data/processed/`:

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

A reduced run saves under the same name as the full run of that variant (only `--quick`
adds a suffix), so running the variant again overwrites it.

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
validation loss and which ones made the ensemble. On Windows, `--live` can crash a run if
the dashboard is reading a progress file at the moment it is replaced, so leave it off for
long unattended runs.

To run every variant in the paper one after another, use `--all`:

```bash
python scripts/03_replicate.py --all --live --device mps
```

For each loss (Sharpe, Gaussian, quantile) that's the baseline plus X-Trend with every
context type and size: CPD 21/63, final hidden state 21/63 and time-equivalent, each
with 10, 20 and 30 contexts. That makes 48 runs. Runs that already have results get
skipped, so if it stops you can run the same command again and it picks up where it
left off. Only the run that was in progress starts over. When it finishes,
`results/replication/tables/sweep_summary.csv` lists every run's metrics next to the
benchmarks'. The other flags (`--seeds`, `--search-iterations`, `--max-steps-per-epoch`,
...) apply to every run in the sweep. Add `--redo` to rerun finished runs.

### 4. Extensions

`scripts/04_extend.py` is a placeholder. The POTENTIAL ideas in it are:

- extending the backtest to 2024-2025 (data already downloaded through 2025; not yet backtested)
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

### 8. Main results table

```bash
python scripts/08_results_table.py --max-steps-per-epoch 300
```

Collects the four main runs (Baseline, X-Trend, X-Trend-G, X-Trend-Q) and the benchmarks
into `results/replication/tables/main_results.md` / `.csv`, plus the hyperparameters each
kept model used in `kept_hyperparameters.csv`. Runs that haven't finished are skipped.
`--max-steps-per-epoch` only labels the table (the cap isn't saved in the training logs).

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
scripts/              the numbered steps above (08 builds the main results table)
notebooks/            exploration and checks on the data pipeline and arrays
results/replication/  tables/ and figures/ from 03_replicate.py and 08_results_table.py
```

## Preliminary Replication Results

> **Note:** these results come from a much smaller training budget than the paper's.
> The code can run all 48 variants with the paper's full search (`03_replicate.py --all`),
> but training that many models on 30 years of data would take weeks on our single GPU.
> To have results before submission, we trained the four main model types with the same
> reduced settings (below). Treat the rankings as indicative, not final.

**Out-of-sample performance, 1995-2023** (49 futures, scaled to 15% annual volatility, before transaction costs)

| Model | Loss | Sharpe<br>1995-2023 | Sharpe<br>2018-2023 | Annual<br>return | Max<br>drawdown |
|:---|:---|---:|---:|---:|---:|
| Baseline (no context) | Sharpe | 1.66 | 1.70 | 24.9% | -31.7% |
| **X-Trend** | Sharpe | **1.86** | **2.28** | **28.0%** | -35.6% |
| X-Trend-G | Gaussian NLL + Sharpe | 0.53 | 0.59 | 8.0% | -93.1% |
| X-Trend-Q | Quantile + Sharpe | 1.76 | 1.90 | 26.4% | -49.2% |
| *Benchmarks* | | | | | |
| Long | - | 0.80 | 0.55 | 12.1% | -49.7% |
| TSMOM | - | 1.12 | 0.47 | 16.7% | -31.0% |
| MACD | - | 0.94 | 0.64 | 14.2% | -31.8% |

X-Trend models use a context set of 10 change-point (CPD) regimes, each up to 21 days long.
Expanding windows train from 1990 and test in 5-year blocks. Best model in bold.

### Training settings

| | Paper | Ours |
|---|---|---|
| Ensemble size per window | 10 | 2 |
| Hyperparameter settings tried per ensemble member | 10 | 1 (random draw) |
| Batches per epoch | all training sequences | at most 300 |
| Max epochs / early stopping | 100 / 10 | 100 / 10 |
| Optimizer | Adam, lr 0.001 | Adam, lr 0.001 |

Hyperparameters are drawn from dropout {0.3, 0.4, 0.5}, hidden size {64, 128}, batch size
{64, 128} and max gradient norm {0.01, 1, 100}. The draws are seeded by window, so all
four models used the same hyperparameters in each window. The kept settings per window are
in `results/replication/tables/kept_hyperparameters.csv`.

Commands (one per model):

```bash
python scripts/03_replicate.py --model baseline --seeds 2 --search-iterations 1 --max-steps-per-epoch 300
python scripts/03_replicate.py --seeds 2 --search-iterations 1 --max-steps-per-epoch 300
python scripts/03_replicate.py --loss gauss --seeds 2 --search-iterations 1 --max-steps-per-epoch 300
python scripts/03_replicate.py --loss qre --seeds 2 --search-iterations 1 --max-steps-per-epoch 300
python scripts/08_results_table.py --max-steps-per-epoch 300
```

### Observations

- **X-Trend beats the no-context baseline and all benchmarks**, most clearly in 2018-2023
  (Sharpe 2.28 vs 1.70 for the baseline and 0.47 for TSMOM). That matches the paper's
  main claim that the context set helps most in recent, regime-changing markets.
- **X-Trend-Q** also beats the baseline, but with a deeper drawdown.
- **X-Trend-G was unstable at this budget.** In the 2005-2010 window its yearly Sharpe
  equals Long's exactly, so the ensemble appears to have collapsed into an always-long
  position, which drove the 2008 loss and the -93% drawdown. With only 2 models per
  window, one failed model can dominate a window.
- All numbers are before transaction costs, and each model is a single run with a small
  ensemble, so close gaps (e.g. baseline 1.66 vs X-Trend-Q 1.76) may not survive a rerun.

### Where the results are

`results/replication/`:

- `tables/main_results.md` / `.csv`: the table above
- `tables/<run>_metrics.csv`: full metrics for each model vs. benchmarks
- `tables/<run>_sharpe_by_year.csv`: Sharpe per calendar year
- `tables/<run>_training_log.csv`: every model trained (settings, validation loss, epochs)
- `figures/<run>_returns.png`: cumulative return, rolling Sharpe and drawdown

Run names: `baseline_sharpe`, `xtrend_sharpe_cpd21_c10` (X-Trend),
`xtrend_gauss_cpd21_c10` (X-Trend-G), `xtrend_qre_cpd21_c10` (X-Trend-Q).
