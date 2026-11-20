"""Experiment 2: Forward-oriented causal observable F(t) with adaptive
derivative mixing, evaluated against F_0 and buy-and-hold on EUR/USD
1-minute data.

Reproduces the forward-shift illustration (paper Figure 2) and the
F(t)-based hysteresis strategy performance against the F_0-based strategy
and buy-and-hold (paper Figure 3 / Table 1, F column).

Run as: python -m exp.exp2 [--no-cache]
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
from src.decision_rule import compounded_equity, hysteresis_state, realized_returns
from src.pipeline import backtest, compute_f, compute_f0, load_price_data, subperiod_report, whole_sample_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_figure2(ohlcv: pd.DataFrame, f0_smoothed: pd.Series, out_path: Path) -> dict:
    """Figure 2 illustration: cumulative returns of the hysteresis rule
    applied to forward-shifted F_0(t+k), k=0..13, versus buy-and-hold, over
    the December 2025 excerpt (exp_2 data_requirements)."""
    mask = (ohlcv["timestamp"] >= pd.Timestamp(config.FIGURE2_WINDOW_START, tz="UTC")) & (
        ohlcv["timestamp"] < pd.Timestamp(config.FIGURE2_WINDOW_END, tz="UTC")
    )
    window_idx = ohlcv.index[mask]
    assert len(window_idx) > 0, "no bars found in the Figure 2 excerpt window"

    returns = ohlcv["ret"]
    end_equity = {}
    fig, ax = plt.subplots(figsize=(11, 5))

    buy_hold_equity = compounded_equity(returns.loc[window_idx].reset_index(drop=True))
    ax.plot(buy_hold_equity.to_numpy(), label="Buy-and-hold", color="black", linewidth=2)
    end_equity["buy_and_hold"] = float(buy_hold_equity.iloc[-1])

    for k in range(config.FORWARD_SHIFT_MAX_K + 1):
        shifted = f0_smoothed.shift(-k)  # F_0(t+k): forward shift, illustration only (Figure 2)
        state = hysteresis_state(shifted, config.DECISION_THETA)
        realized = realized_returns(state, returns)
        window_equity = compounded_equity(realized.loc[window_idx].reset_index(drop=True))
        label = "F_0(t)" if k == 0 else f"F_0(t+{k})"
        ax.plot(window_equity.to_numpy(), label=label, alpha=0.8)
        end_equity[f"f0_shift_{k}"] = float(window_equity.iloc[-1])

    ax.set_title("Figure 2 (paper Sec. 2.3): forward-shifted F_0(t+k) vs. buy-and-hold")
    ax.set_xlabel("Bar index within excerpt")
    ax.set_ylabel("Compounded equity (V_0 = 1 at excerpt start)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return end_equity


def make_equity_comparison_plot(
    timestamps: pd.Series, equity_f: pd.Series, equity_f0: pd.Series, buy_hold: pd.Series, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(timestamps, equity_f.to_numpy(), label="F(t) forward-oriented strategy", color="tab:red")
    ax.plot(timestamps, equity_f0.to_numpy(), label="F_0(t) strategy", color="tab:blue")
    ax.plot(timestamps, buy_hold.to_numpy(), label="Buy-and-hold", color="tab:gray")
    ax.set_title("Figure 3 (paper Sec. 3): compounded equity, F vs. F_0 vs. buy-and-hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Compounded equity V_t (V_0 = 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(start: str, end: str, use_cache: bool = True) -> dict:
    """Run the full exp_2 pipeline and return a JSON-serializable results dict."""
    t0 = time.time()
    ohlcv = load_price_data(start, end, use_cache=use_cache)
    n_bars = len(ohlcv)
    assert n_bars > 0, "no bars survived the active Forex session filter"

    f0_frame = compute_f0(ohlcv)
    f_series = compute_f(f0_frame["f0"])

    result_f = backtest(ohlcv, f_series, config.DECISION_THETA)
    result_f0 = backtest(ohlcv, f0_frame["f0"], config.DECISION_THETA)

    whole_f = whole_sample_report(result_f["equity"])
    whole_f0 = whole_sample_report(result_f0["equity"])
    buy_hold_whole = whole_sample_report(result_f["buy_hold_equity"])

    splits = [
        ("2023-Sep2024", pd.Timestamp(start, tz="UTC"), pd.Timestamp(config.SUBPERIOD_1_END, tz="UTC")),
        ("post-Sep2024", pd.Timestamp(config.SUBPERIOD_1_END, tz="UTC"), pd.Timestamp(end, tz="UTC")),
    ]
    sub_report_f = subperiod_report(ohlcv["timestamp"], result_f["realized"], result_f["state"], splits)
    sub_report_f0 = subperiod_report(ohlcv["timestamp"], result_f0["realized"], result_f0["state"], splits)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sub_report_f.to_csv(RESULTS_DIR / "exp2_subperiod_report_F.csv", index=False)
    sub_report_f0.to_csv(RESULTS_DIR / "exp2_subperiod_report_F0.csv", index=False)
    make_equity_comparison_plot(
        ohlcv["timestamp"], result_f["equity"], result_f0["equity"], result_f["buy_hold_equity"],
        RESULTS_DIR / "exp2_equity_curve.png",
    )
    figure2_end_equity = make_figure2(ohlcv, f0_frame["f0"], RESULTS_DIR / "exp2_figure2_forward_shift.png")

    summary = {
        "n_bars": int(n_bars),
        "date_range": [str(ohlcv["timestamp"].min()), str(ohlcv["timestamp"].max())],
        "runtime_seconds": time.time() - t0,
        "f_strategy": whole_f,
        "f0_strategy": whole_f0,
        "buy_and_hold": buy_hold_whole,
        "total_trades_f": float(result_f["activity"].iloc[-1]),
        "total_trades_f0": float(result_f0["activity"].iloc[-1]),
        "subperiods_F": sub_report_f.to_dict(orient="records"),
        "subperiods_F0": sub_report_f0.to_dict(orient="records"),
        "figure2_end_equity": figure2_end_equity,
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as handle:
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
