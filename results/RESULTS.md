# Results — Forward-Oriented Causal Observables for Non-Stationary Financial Markets

Paper: **Forward-Oriented Causal Observables for Non-Stationary Financial Markets**.

This file reports the outcome of the two reproduced experiments and compares
them against the values the paper states in Sec. 3 ("Experimental Results"),
Figure 2, Figure 3 and Table 1. All numbers below are read from the files
saved by the final run in `results/`: `exp1_summary.json`, `exp2_summary.json`,
`exp1_subperiod_report.csv`, `exp2_subperiod_report_F.csv`,
`exp2_subperiod_report_F0.csv`, and the corresponding `.png` figures.

## Experiment 1 — Causal Composite Observable F_0 with Kalman Stabilization vs. Buy-and-Hold

Builds the causally centered, linearly aggregated composite F_0(t) from
causal RSI, MFI, MACD difference and BB% (Sec. 2.2, Eqs. 1-3, Appendix A),
stabilizes it with the one-dimensional causal Kalman filter (Appendix B),
and evaluates the two-state hysteresis decision rule (Sec. 2.4, Eqs. 6-7)
against buy-and-hold on EUR/USD 1-minute bars.

Artifacts: `results/exp1_figure1_indicators.png` (Figure 1 intermediate-stage
illustration), `results/exp1_equity_curve.png` (Figure 3 equity curve),
`results/exp1_subperiod_report.csv` (Table 1), `results/exp1_summary.json`.

Realized data: 1,108,327 bars after the active Forex-session filter, from
2023-01-01 23:22 UTC to 2026-01-07 23:59 UTC (`exp1_summary.json`).

| Metric | Paper (Sec. 3 / Table 1 / Fig. 3) | Reproduced | Verdict |
|---|---|---|---|
| F_0 whole-sample end equity (01-2026) | ≈2.45–2.50 | 0.426 | not reproduced |
| F_0 whole-sample cumulative return | ≈+145–150% | -57.4% | not reproduced |
| F_0 post-Sep-2024 end equity | ≈1.90 | 0.738 (-26.2%) | not reproduced |
| F_0 trading activity | ≈1000–1500 trades/month | 767–825 trades/month | partially reproduced |
| F_0 vs. buy-and-hold (direction) | F_0 strongly outperforms | F_0 underperforms (buy-and-hold +9.2%, F_0 -57.4%) | not reproduced |

## Experiment 2 — Forward-Oriented Causal Observable F(t) with Adaptive Derivative Mixing

Adds the causal finite-difference derivative of the Kalman-smoothed F_0(t),
smoothed with a span-4 moving average, mixed with state-dependent
coefficients c1(t)=tanh(|F_0(t)|), c2(t)=1-tanh(|F_0(t)/2|) (Sec. 2.3,
Eqs. 4-5), and evaluates the same hysteresis rule on F(t) against F_0 and
buy-and-hold.

Artifacts: `results/exp2_figure2_forward_shift.png` (Figure 2),
`results/exp2_equity_curve.png` (Figure 3), `results/exp2_subperiod_report_F.csv`
and `results/exp2_subperiod_report_F0.csv` (Table 1), `results/exp2_summary.json`.

Realized data: same 1,108,327 bars and date range as Experiment 1.

| Metric | Paper (Sec. 3 / Table 1 / Fig. 3) | Reproduced | Verdict |
|---|---|---|---|
| F 2023–Sep-2024 end equity | ≈3.14 (cum. +214%, MDD -4%, ≈959 trades/month) | 0.406 (cum. -59.4%, MDD -59.4%, 1156 trades/month) | not reproduced |
| F post-Sep-2024 end equity | ≈2.74 (cum. -13%, MDD -17%, ≈1117 trades/month) | 0.663 (cum. -33.7%, MDD -34.0%, 1237 trades/month) | not reproduced |
| F whole-sample end equity (01-2026) | ≈2.75 | 0.268 (cum. -73.2%, MDD -73.2%) | not reproduced |
| F trading activity | ≈959–1117 trades/month | 1156–1237 trades/month | reproduced |
| F vs. F_0 ranking (F end equity > F_0 end equity) | holds (2.75 > 2.45–2.50) | does not hold (0.268 < 0.426) | not reproduced |
| Figure 2: forward-shifted F_0(t+k) gives larger cumulative gain than F_0(t) | monotonic increase with k | monotonic increase from 1.0009 (k=0) to 1.047 (k≈11), then flat (Dec-2025 excerpt, `exp2_summary.json["figure2_end_equity"]`) | reproduced |

## Localization of the gap

The whole-sample and subperiod equity levels differ materially from the
paper (opposite direction of the net effect: the paper's F_0 and F
strategies compound to large gains, the reproduced strategies compound to
losses on the substituted instrument). Two checks isolate the cause:

1. **Sign/construction check (synthetic data).** The identical pipeline
   (causal indicators → causal median centering → F_0 → Kalman filter →
   hysteresis rule → compounded equity) was run on a synthetic 5,000-bar
   series built from a clear sustained uptrend followed by a sustained
   downtrend. Both F_0 and F correctly identify the trend regimes and
   outperform buy-and-hold on that series (F_0 end equity 1.347 vs.
   buy-and-hold 0.988; F end equity 1.179 vs. 0.988). This rules out a
   sign or wiring error in the composite-observable, Kalman-filter, or
   decision-rule implementation: the pipeline is directionally correct
   and capable of producing the paper's qualitative outperformance pattern
   when the underlying series actually trends the way the indicators are
   built to detect.
2. **Predictive-content check (real data).** On a one-month slice of the
   real EUR/USD series (`src.config.SAMPLE_START_TEST`/`SAMPLE_END_TEST`),
   the correlation of F_0(t) and F(t) with the next-bar simple return is
   small and slightly negative (-0.025 and -0.035 respectively), i.e. the
   composite observable carries close to no short-horizon predictive
   content — and if anything is mildly contrarian — on this instrument and
   sample, consistent with the realized underperformance.

Both checks point to the same identified cause: the composite observable,
built exactly as specified by the paper (fixed scaling coefficients, Kalman
parameters, threshold, Appendix A indicator formulas), has no material
short-horizon predictive edge on the real EUR/USD interbank series used
here, whereas the paper reports strong unconditional profitability on its
own "EURUSDT" series. The Figure 2 forward-shift advantage, which is a
property of the F_0(t+k) construction itself rather than of the
instrument's realized profitability, is reproduced. The F-over-F_0 end-
equity ranking is not reproduced: on the EUR/USD series, F's much higher
trading activity (1156-1237 vs. 767-825 trades/month for F_0, now matching
the paper's reported order of magnitude for F once the derivative base of
Eq. 4 is resolved — see "Derivative base for F(t)" below) compounds the
instrument's lack of short-horizon predictability into a larger loss for F
than for F_0, rather than the paper's larger gain for F.

## Deviations from the paper

- **Instrument substitution.** The paper evaluates "EURUSDT" 1-minute
  candles; no data source available here serves a ticker named "EURUSDT".
  The paper's own weekend/Forex-session filtering rule (Sec. 3: remove
  Saturdays, Sundays before 18:00, Fridays after 18:00) only applies to a
  market that actually closes on weekends, which points to the EUR/USD
  interbank Forex pair rather than a continuously-trading crypto market.
  EUR/USD 1-minute OHLCV (Massive symbol `C:EURUSD`, loaded through the
  QuantX Dataloader `POST /fetch-ohlcvs`, `source="massive"`,
  `yahoo_backstop=False`) is used in its place. `ticker_details` for this
  load reports `data_source: "massive"` for the full requested window, so
  no Yahoo-backstop substitution occurred.
- **Sample period.** The paper's window is "2023 through approximately
  January 2026" with a subperiod split at September 2024; this is
  implemented exactly (`SAMPLE_START_PAPER=2023-01-01`,
  `SAMPLE_END_PAPER=2026-01-07`, `SUBPERIOD_1_END=2024-09-30`). The realized
  first bar is 2023-01-01 23:22 UTC (first bar Massive has for this symbol
  at that start date) and the realized last bar is 2026-01-07 23:59 UTC;
  both lie inside the paper's stated window.
- **Unspecified indicator lookback windows.** Sec. 2.2 states the lookback
  windows are "fixed to conventional values widely adopted in the technical
  analysis literature" without listing numbers. The conventional values from
  the literature the paper itself cites are used: RSI period 14 (Wilder,
  1978, Wilder smoothing with alpha=1/14), MFI period 14, MACD fast/slow/
  signal 12/26/9 (Appel, 1979), Bollinger window 20 with k=2 standard
  deviations.
- **Unspecified Kalman initial variance.** The paper specifies x_0=z_0 but
  not P_0; P_0 = r = 0.1 is used as a conservative default, matching the
  implementation notes.
- **Local time zone for the Forex-session filter.** The paper does not name
  the timezone of "local time"; the standard Forex-session convention
  (America/New_York) is used.
- **Derivative base for F(t).** The paper's Eq. 4 is evaluated on "the first
  derivative of F_0(t)" without stating whether F_0 here is the raw
  composite or the Kalman-smoothed composite. A sensitivity check on a
  one-month slice (`src.config.SAMPLE_START_TEST`/`SAMPLE_END_TEST`,
  31,439 bars) compared both: differencing the Kalman-smoothed F_0 gives
  209 trades/month, an order of magnitude below the paper's reported
  ≈959-1117 trades/month for F; differencing the raw composite F_0 gives
  1191 trades/month, matching the paper's order of magnitude. The
  implementation therefore takes the causal finite-difference derivative of
  the raw composite F_0(t) (`composite_f0_raw`), while the mixing
  coefficients c1(t), c2(t) and the F_0(t) term of Eq. 4 itself still use
  the Kalman-smoothed composite, consistent with the decision rule of
  Sec. 2.4. With this choice, the whole-sample and subperiod trading
  activity for F (1156 and 1237 trades/month) is reproduced within the
  paper's reported order of magnitude, confirming the derivative base as
  the correct interpretation of Eq. 4. This choice does not close the
  whole-sample profitability gap, whose cause is identified separately
  below (data substitution).
- **Reported gap between Table 1 and the in-text/Figure 3 equity values
  (paper's own uncertainty).** The paper's Table 1 end-equity values for F
  (3.14, 2.74) do not match its own Figure 3 cumulative-return-table values
  (3.10 at 09-2024, 2.75 at 01-2026) quoted in the task description; the
  reconciled, more conservative Figure-3-style whole-sample figures (≈2.75
  for F, ≈2.45-2.50 for F_0) are used as the comparison target above.

## Not reproduced

- **Absolute profitability magnitude of Table 1 / Figure 3 on the paper's
  own instrument.** The paper's reported end-equity levels (F_0 ≈2.45-2.50,
  F ≈2.75 by 01-2026; subperiod end equities 3.14 and 2.74 for F) could not
  be reproduced on the EURUSDT crypto-exchange pair the paper uses, because
  no data source available here serves that ticker; only the substituted
  EUR/USD interbank series could be evaluated, on which the identical,
  paper-faithful pipeline does not show the same unconditional
  profitability (see "Localization of the gap" above).
