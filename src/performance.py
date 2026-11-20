"""Performance metrics and subperiod reporting (Sec. 3, Table 1).

# Custom implementation: end equity, cumulative return, and max drawdown are
# generic quantstats/empyrical metrics, but the paper's subperiod-normalized
# equity (V_0 = 1 reset at the start of each regime, Table 1 caption) and
# trades-per-month from a cumulative trade count are specific reporting
# conventions implemented directly on the equity/activity series this
# pipeline already produces.
"""

import numpy as np
import pandas as pd


def max_drawdown_pct(equity: pd.Series) -> float:
    """Maximum peak-to-trough percentage decline of an equity curve."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min() * 100.0)


def trades_per_month(activity: pd.Series, index: pd.DatetimeIndex) -> float:
    """Average number of state changes per calendar month.

    ``activity`` is the cumulative trading-activity series (cumsum of
    |Delta p_t|); the total number of trades over the window divided by the
    number of elapsed months gives trades per month.
    """
    if len(activity) == 0:
        return float("nan")
    total_trades = float(activity.iloc[-1] - activity.iloc[0])
    elapsed_days = (index[-1] - index[0]).total_seconds() / 86400.0
    elapsed_months = max(elapsed_days / 30.436875, 1e-9)  # average Gregorian month
    return total_trades / elapsed_months


def subperiod_metrics(
    timestamps: pd.Series,
    realized_returns: pd.Series,
    state: pd.Series,
    period_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """End equity, cumulative return %, max drawdown %, and trades/month for
    one subperiod window, with equity re-normalized to V_0 = 1 at the start
    of the window (Table 1 caption: "cumulative return curve normalized to
    V_0 = 1 at the beginning of each regime").
    """
    mask = (timestamps >= start) & (timestamps < end)
    if mask.sum() == 0:
        return {
            "period": period_label,
            "end_equity": float("nan"),
            "cum_return_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
            "trades_per_month": float("nan"),
            "n_bars": 0,
        }
    window_returns = realized_returns[mask]
    window_state = state[mask]
    window_index = timestamps[mask]

    equity = (1.0 + window_returns.fillna(0.0)).cumprod()
    activity = window_state.diff().abs().fillna(0.0).cumsum()

    return {
        "period": period_label,
        "end_equity": float(equity.iloc[-1]),
        "cum_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "trades_per_month": trades_per_month(activity, pd.DatetimeIndex(window_index)),
        "n_bars": int(mask.sum()),
    }


def summary_table(equity: pd.Series) -> dict:
    """Whole-sample end equity, cumulative return %, and max drawdown %."""
    if len(equity) == 0:
        return {"end_equity": float("nan"), "cum_return_pct": float("nan"),
                "max_drawdown_pct": float("nan")}
    return {
        "end_equity": float(equity.iloc[-1]),
        "cum_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct(equity),
    }
