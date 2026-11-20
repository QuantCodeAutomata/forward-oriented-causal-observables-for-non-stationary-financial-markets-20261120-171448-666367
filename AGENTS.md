# Repository guide

This repository reproduces two experiments from "Forward-Oriented Causal
Observables for Non-Stationary Financial Markets" (Sec. 2-3): the causal
composite observable F_0(t) with Kalman stabilization (Experiment 1), and
the forward-oriented observable F(t) with adaptive derivative mixing
(Experiment 2), both evaluated with a two-state hysteresis decision rule on
EUR/USD 1-minute bars.

## Layout

- `src/` — flat library modules: `indicators.py` (causal RSI/MFI/MACD/BB%,
  Appendix A), `signal_construction.py` (causal centering, F_0, Kalman
  filter, forward operator, Sec. 2.2-2.3), `decision_rule.py` (hysteresis
  state, causal execution, compounded equity, Sec. 2.4), `performance.py`
  (drawdown, trades/month, subperiod reporting, Table 1), `forex_session.py`
  (active-hours filter, Sec. 3), `pipeline.py` (the one shared data-load /
  signal / backtest path both experiments call), `config.py` (every
  parameter, each commented with its paper section or "not specified in the
  paper").
- `exp/` — `exp1.py` and `exp2.py`, runnable as `python -m exp.exp1` /
  `python -m exp.exp2`; they only call `src.pipeline` functions, never a
  private loader or shortcut.
- `data/data_loader.py` — EUR/USD OHLCV bars via the QuantX Dataloader REST
  API (`POST https://quantx-api.limex.pro/apis/dataloader/fetch-ohlcvs`);
  downloaded bars are cached under `data/cache/` (git-ignored) and never
  committed.
- `tests/` — unit tests for every `src/` module on synthetic fixtures, plus
  `tests/test_pipeline_market_data.py`, marked `market_data` and skipped when
  `DATALOADER_API_KEY`/`MASSIVE_API_KEY` are absent, which checks the shared
  pipeline on a short real-data slice cached via `request.config.cache`.
- `results/` — every generated table, plot and summary file, plus
  `results/RESULTS.md` with the comparison to the paper and the deviations.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # full suite, requires API keys

python -m exp.exp1   # Experiment 1: F_0 vs. buy-and-hold
python -m exp.exp2   # Experiment 2: F vs. F_0 vs. buy-and-hold

bash reproduce.sh    # tests + both experiments
```

## Conventions

- Every experiment parameter is a module-level variable in `src/config.py`;
  no parameters are hardcoded in `src/` or `exp/`.
- Market data is loaded only through `data/data_loader.py`; no other module
  calls an external data API.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from an uncommitted
  `.env` file (via `python-dotenv`); they never appear in code, logs, or
  results.
- Running `python -m exp.exp1` / `python -m exp.exp2` overwrites the files
  under `results/`. After a rerun, the numbers and verdicts in
  `results/RESULTS.md` must be checked against the newly written files and
  updated by hand; they are not regenerated automatically.
