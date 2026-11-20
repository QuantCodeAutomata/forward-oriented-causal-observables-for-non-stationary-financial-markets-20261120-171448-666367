"""Experiment 1: Causal composite observable F_0 with Kalman stabilization,
evaluated against buy-and-hold on EUR/USD 1-minute data.

Reproduces the intermediate-stage illustration (paper Figure 1) and the
performance evaluation of the F_0-based hysteresis strategy against
buy-and-hold (paper Figure 3 / Table 1, F_0 column), without the
forward-oriented derivative operator of Sec. 2.3.

Run as: python -m exp.exp1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.pipeline import backtest, compute_f0, load_price_data, subperiod_report, whole_sample_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_figure1(ohlcv: pd.DataFrame, f0_frame: pd.DataFrame, out_path: Path) -> None:
    """Figure 1 illustration: the four causal indicators and the raw vs.
    Kalman-smoothed composite F_0(t) on a short excerpt (exp_1 data
    requirements: "approximately 200 1-minute candles")."""
    valid = f0_frame.dropna(subset=["f0"])
    start_idx = valid.index[0] if len(valid) else 0
    excerpt = f0_frame.loc[start_idx:].iloc[: config.FIGURE1_NUM_CANDLES]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(excerpt.index, excerpt["rsi"], label="RSI", color="tab:blue")
    axes[0].plot(excerpt.index, excerpt["mfi"], label="MFI", color="tab:orange")
    axes[0].set_title("Causal RSI and MFI")
    axes[0].set_ylabel("Index value")
    axes[0].legend()

    axes[1].plot(excerpt.index, excerpt["macd_diff"], label="MACD difference", color="tab:green")
    ax2 = axes[1].twinx()
    ax2.plot(excerpt.index, excerpt["bb_pct"], label="BB%", color="tab:red")
    axes[1].set_title("Causal MACD difference and Bollinger %B")
    axes[1].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes[2].plot(excerpt.index, excerpt["f0_raw"], label="F_0 raw (Eq. 3)", color="gray", alpha=0.6)
    axes[2].plot(excerpt.index, excerpt["f0"], label="F_0 Kalman-smoothed (Appendix B)", color="black")
    axes[2].axhline(config.DECISION_THETA, color="green", linestyle="--", linewidth=0.8, label="+theta")
    axes[2].axhline(-config.DECISION_THETA, color="red", linestyle="--", linewidth=0.8, label="-theta")
    axes[2].set_title("Raw vs. Kalman-stabilized composite F_0(t)")
    axes[2].set_xlabel("Bar index")
    axes[2].legend()

    fig.suptitle("Figure 1 (paper Sec. 2.2): intermediate-stage causal signal construction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_equity_plot(timestamps: pd.Series, equity: pd.Series, buy_hold: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(timestamps, equity.to_numpy(), label="F_0 Kalman-smoothed strategy", color="tab:blue")
    ax.plot(timestamps, buy_hold.to_numpy(), label="Buy-and-hold", color="tab:gray")
    ax.set_title("Figure 3 (paper Sec. 3): compounded equity, F_0 strategy vs. buy-and-hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Compounded equity V_t (V_0 = 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(start: str, end: str, use_cache: bool = True) -> dict:
    """Run the full exp_1 pipeline and return a JSON-serializable results dict."""
    t0 = time.time()
    ohlcv = load_price_data(start, end, use_cache=use_cache)
    n_bars = len(ohlcv)
    assert n_bars > 0, "no bars survived the active Forex session filter"

    f0_frame = compute_f0(ohlcv)
    result = backtest(ohlcv, f0_frame["f0"], config.DECISION_THETA)

    whole = whole_sample_report(result["equity"])
    buy_hold_whole = whole_sample_report(result["buy_hold_equity"])

    splits = [
        ("2023-Sep2024", pd.Timestamp(start, tz="UTC"), pd.Timestamp(config.SUBPERIOD_1_END, tz="UTC")),
        ("post-Sep2024", pd.Timestamp(config.SUBPERIOD_1_END, tz="UTC"), pd.Timestamp(end, tz="UTC")),
    ]
    sub_report = subperiod_report(ohlcv["timestamp"], result["realized"], result["state"], splits)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sub_report.to_csv(RESULTS_DIR / "exp1_subperiod_report.csv", index=False)
    make_figure1(ohlcv, f0_frame, RESULTS_DIR / "exp1_figure1_indicators.png")
    make_equity_plot(ohlcv["timestamp"], result["equity"], result["buy_hold_equity"], RESULTS_DIR / "exp1_equity_curve.png")

    summary = {
        "n_bars": int(n_bars),
        "date_range": [str(ohlcv["timestamp"].min()), str(ohlcv["timestamp"].max())],
        "runtime_seconds": time.time() - t0,
        "f0_strategy": whole,
        "buy_and_hold": buy_hold_whole,
        "total_trades": float(result["activity"].iloc[-1]),
        "subperiods": sub_report.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="Bypass the on-disk data cache.")
    args = parser.parse_args()

    summary = run(config.SAMPLE_START_PAPER, config.SAMPLE_END_PAPER, use_cache=not args.no_cache)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
