# Forward-Oriented Causal Observables for Non-Stationary Financial Markets

Reproduction of two experiments from **"Forward-Oriented Causal Observables
for Non-Stationary Financial Markets"**, Sec. 2-3:

1. **Experiment 1** — the causally centered, linearly aggregated composite
   observable F_0(t) built from four technical indicators (RSI, MFI, MACD
   difference, BB%), stabilized with a one-dimensional causal Kalman filter,
   evaluated with a two-state hysteresis decision rule against buy-and-hold.
2. **Experiment 2** — the forward-oriented observable F(t), which adds a
   causal, adaptively-mixed finite-difference derivative term to F_0(t),
   evaluated against the F_0-based rule and buy-and-hold.

Both experiments use EUR/USD 1-minute OHLCV bars (the paper's "EURUSDT"
series; see "Deviations from the paper" in `results/RESULTS.md` for the
instrument substitution), 2023-01-01 through approximately 2026-01-07,
restricted to the paper's active Forex trading-hour filter.

## Repository layout

```
src/        causal indicators, signal construction, decision rule, performance
            metrics, and the shared data/signal/backtest pipeline
exp/        exp1.py (F_0 vs. buy-and-hold), exp2.py (F vs. F_0 vs. buy-and-hold)
data/       data_loader.py -- OHLCV bars via the QuantX Dataloader REST API
tests/      unit tests for every src/ module, plus a market-data test
results/    RESULTS.md and every generated table, plot, and summary file
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"      # fast unit tests, no network access
python -m pytest                           # full suite, requires API keys
```

## Experiments

```bash
python -m exp.exp1   # Experiment 1: F_0 vs. buy-and-hold
python -m exp.exp2   # Experiment 2: F vs. F_0 vs. buy-and-hold
```

Both scripts load the full 2023-01-01 to 2026-01-07 EUR/USD 1-minute window
(`src/config.py`: `SAMPLE_START_PAPER`, `SAMPLE_END_PAPER`), which is the
paper's own evaluation window; there is no reduced-universe mode for these
two single-instrument experiments. Results are written to `results/` and
overwrite any previous run of the same experiment.

## Results

See `results/RESULTS.md` for the full comparison against the paper's
reported values (Sec. 3, Figures 1-3, Table 1), the localization of the
reproduced performance gap, and every deviation from the paper.

## Reproduce everything

```bash
bash reproduce.sh
```
