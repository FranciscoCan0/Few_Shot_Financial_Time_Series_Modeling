"""Step 4: extensions beyond the paper (not decided yet).

Candidates from the project outline:
  - extend the data to 2024-2026: set end_date in configs/data.json and backtest.test_end
    in configs/model.json, rerun scripts 01 and 02 (CPD takes ~3 h), then score the
    trained setup on 2024-2026 with project.training and project.evaluation;
  - transaction costs: evaluation.portfolio_returns(pred, cost=C) with C > 0 (Eq. 2);
  - select the 50 futures by liquidity at the start of the sample, not the end.
"""

if __name__ == "__main__":
    raise SystemExit("04_extend.py is a placeholder: no extension has been chosen yet.")
