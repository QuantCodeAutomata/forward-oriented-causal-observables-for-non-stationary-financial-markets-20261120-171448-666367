#!/usr/bin/env bash
# Reproduce the causal-observable experiments end to end.
set -uo pipefail

python -m pytest -m "not market_data"
python -m exp.exp1
python -m exp.exp2
