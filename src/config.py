"""Parameters for the causal-observable replication (paper: 'Forward-Oriented
Causal Observables for Non-Stationary Financial Markets').

Every experiment parameter lives here as a module-level variable. Each
variable is commented with its meaning, unit, and the paper section or
equation it comes from; "not specified in the paper" marks values the paper
leaves open, for which a conventional literature default is used.
"""

# ---------------------------------------------------------------------------
# Instrument and sample period (Sec. 3 "Experimental Results";
# data_requirements of exp_1/exp_2)
# ---------------------------------------------------------------------------
# Massive reference data has no crypto ticker "EURUSDT"; the paper's own
# weekend/Forex-session filtering (Sec. 3) only makes sense for a market that
# closes on weekends, so the forex pair EUR/USD (Massive symbol "C:EURUSD")
# is used in place of "EURUSDT" (documented in results/RESULTS.md).
TICKER = "C:EURUSD"  # Massive REST symbol for the EUR/USD exchange rate
TIMEFRAME = "1m"  # bar frequency, Sec. 3: "1-minute frequency"
SAMPLE_START_PAPER = "2023-01-01"  # paper: "2023" start of evaluation window
SAMPLE_END_PAPER = "2026-01-07"  # paper Figure 3/Table 1: last point plotted is 01-2026
# Reduced period used for the fast unit-test fixture (tests/conftest.py), not
# for the reported experiment results.
SAMPLE_START_TEST = "2024-01-02"
SAMPLE_END_TEST = "2024-01-31"

# Subperiod split for Table 1 (Sec. 3 / Table 1 caption)
SUBPERIOD_1_END = "2024-09-30"  # "2023-Sep 2024"
# the second subperiod ("Post-Sep 2024") runs from SUBPERIOD_1_END to the sample end

# Forex trading-hour filter (Sec. 3, paragraph before Figure 3): "Saturdays are
# removed entirely, as well as Sundays before 18:00 and Fridays after 18:00
# (local time)". The paper does not name the timezone of "local time"; the
# standard Forex-session convention (New York close/open at 17:00-18:00) is
# used, i.e. "local time" = America/New_York (not specified in the paper).
FOREX_SESSION_TZ = "America/New_York"
FOREX_SESSION_CUTOFF_HOUR = 18  # Sec. 3: "18:00"

# ---------------------------------------------------------------------------
# Technical indicators (Appendix A). Sec. 2.2 states lookback windows are
# "fixed to conventional values widely adopted in the technical analysis
# literature", without listing numbers -- not specified in the paper. The
# standard values from the works the paper itself cites for each indicator
# are used (Wilder 1978 for RSI/MFI, Appel 1979 for MACD, Bollinger for BB%).
# ---------------------------------------------------------------------------
RSI_PERIOD = 14  # bars; Wilder (1978) conventional RSI lookback
MFI_PERIOD = 14  # bars; conventional MFI lookback (same convention as RSI)
MACD_FAST_PERIOD = 12  # bars; Appel (1979) conventional fast EMA
MACD_SLOW_PERIOD = 26  # bars; Appel (1979) conventional slow EMA
MACD_SIGNAL_PERIOD = 9  # bars; Appel (1979) conventional signal EMA
BB_WINDOW = 20  # bars; conventional Bollinger Band lookback
BB_NUM_STD = 2.0  # conventional Bollinger Band width (k), Eq. A.4

# ---------------------------------------------------------------------------
# Composite signal F_0(t): causal centering and linear aggregation
# (Sec. 2.2, Eqs. 1-3; scaling coefficients from Figure 1)
# ---------------------------------------------------------------------------
ALPHA_MFI = 1.0 / 200.0  # Fig. 1 scaling coefficient for MFI
ALPHA_RSI = 1.0 / 100.0  # Fig. 1 scaling coefficient for RSI
ALPHA_BB_PCT = 1.0 / 3.0  # Fig. 1 scaling coefficient for BB%
ALPHA_MACD = 1.0 / 3e-4  # Fig. 1 scaling coefficient for MACD difference (~3333.33)

# ---------------------------------------------------------------------------
# Kalman stabilization (Sec. 2.2, Appendix B, Eqs. B.1-B.2)
# ---------------------------------------------------------------------------
KALMAN_Q = 0.01  # process-noise variance, Sec. 2.2: "q = 0.01"
KALMAN_R = 0.1  # measurement-noise variance, Sec. 2.2: "r = 0.1"
# Initial state covariance P_0: the paper only states x_0 = z_0 (Appendix B)
# and does not specify P_0 -- not specified in the paper; P_0 = r is used as a
# conservative default (the measurement-noise variance), as suggested by the
# accompanying implementation notes.
KALMAN_P0 = KALMAN_R

# ---------------------------------------------------------------------------
# Forward-oriented operator F(t) (Sec. 2.3, Eqs. 4-5)
# ---------------------------------------------------------------------------
DERIVATIVE_SMOOTHING_SPAN = 4  # Eq. 4: "moving average with span 4"
FORWARD_DERIVATIVE_SCALE = 2.0  # Eq. 4: fixed factor multiplying the derivative term

# ---------------------------------------------------------------------------
# Decision rule and execution (Sec. 2.4, Eqs. 6-7)
# ---------------------------------------------------------------------------
DECISION_THETA = 0.06  # Sec. 2.4 / 3: hysteresis threshold theta
EXECUTION_DELAY = 1  # bars; Sec. 2.4: realized state at t is p_{t-1}
TRANSACTION_COSTS = 0.0  # Sec. 3: "all results ... computed without transaction costs"

# ---------------------------------------------------------------------------
# Figure 2 forward-shift illustration (exp_2 only)
# ---------------------------------------------------------------------------
FORWARD_SHIFT_MAX_K = 13  # exp_2 description: "F_0(t+1), ..., F_0(t+13)"
FIGURE2_WINDOW_START = "2025-12-01"  # exp_2 data_requirements: "01-12-2025 to 06-12-2025"
FIGURE2_WINDOW_END = "2025-12-06"

# ---------------------------------------------------------------------------
# Figure 1 intermediate-stage illustration excerpt (exp_1 only)
# ---------------------------------------------------------------------------
FIGURE1_NUM_CANDLES = 200  # exp_1 data_requirements: "approximately 200 1-minute candles"
