## Main results

| Model | Loss | Context set | Sharpe 1995-2023 | Sharpe 2018-2023 | Annual return | Max drawdown |
|---|---|---|---|---|---|---|
| Baseline (no context) | Sharpe | none | 1.66 | 1.70 | 24.9% | -31.7% |
| X-Trend | Sharpe | CPD regimes, max 21 days, 10 contexts | 1.86 | 2.28 | 28.0% | -35.6% |
| X-Trend-G | Gaussian NLL + Sharpe | CPD regimes, max 21 days, 10 contexts | 0.53 | 0.59 | 8.0% | -93.1% |
| X-Trend-Q | Quantile + Sharpe | CPD regimes, max 21 days, 10 contexts | 1.76 | 1.90 | 26.4% | -49.2% |
| Long | - | - | 0.80 | 0.55 | 12.1% | -49.7% |
| TSMOM | - | - | 1.12 | 0.47 | 16.7% | -31.0% |
| MACD | - | - | 0.94 | 0.64 | 14.2% | -31.8% |

## Training settings and kept hyperparameters

| Model | Ensemble size | Settings tried per member | Mean epochs | Kept hidden_size | Kept dropout | Kept batch_size | Kept max_grad_norm |
|---|---|---|---|---|---|---|---|
| Baseline (no context) | 2 | 1 | 20.9 | 64: 8, 128: 4 | 0.4: 6, 0.5: 4, 0.3: 2 | 64: 9, 128: 3 | 1: 5, 100: 4, 0.01: 3 |
| X-Trend | 2 | 1 | 20.2 | 64: 8, 128: 4 | 0.4: 6, 0.5: 4, 0.3: 2 | 64: 9, 128: 3 | 1: 5, 100: 4, 0.01: 3 |
| X-Trend-G | 2 | 1 | 41.5 | 64: 8, 128: 4 | 0.4: 6, 0.5: 4, 0.3: 2 | 64: 9, 128: 3 | 1: 5, 100: 4, 0.01: 3 |
| X-Trend-Q | 2 | 1 | 24.3 | 64: 8, 128: 4 | 0.4: 6, 0.5: 4, 0.3: 2 | 64: 9, 128: 3 | 1: 5, 100: 4, 0.01: 3 |

Returns scaled to 15% annual volatility, transaction cost 0. Expanding windows: train from 1990, test in 5-year blocks from 1995 to 2023. Adam, learning rate 0.001, at most 100 epochs, early stopping after 10 epochs without a better validation loss, batches per epoch capped at: 300. Paper setting: ensemble of 10, 10 settings tried per member. Kept hyperparameters are value: number of kept models over all windows; search space {'dropout': [0.3, 0.4, 0.5], 'hidden_size': [64, 128], 'batch_size': [64, 128], 'max_grad_norm': [0.01, 1.0, 100.0]}.
