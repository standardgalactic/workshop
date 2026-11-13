#!/bin/sh
# Reproduces the three sets of runs reported in the essay. Requires Python 3 and PyTorch (CPU is sufficient).
# Run time is roughly 3 minutes for the first set, 10 minutes for the second, and 10 to 15 minutes for the third on two CPU cores.
set -e

# 0. Exact check of the scalar results (standard library only).
python3 check_scalar.py
mkdir -p results/sweep results/diagnostic

# 1. Synchronization intervals H = 50, 200, 500 on the modeled LAN and WAN, plain averaging and Nesterov (lr 0.7, momentum 0.9).
python3 round_length.py --seeds 2 --budget 3000 --out results/round_length_results.json

# 2. Outer optimizer sweep at H = 500 on the WAN link.
for lr in 0.1 0.3 0.7; do
  for mu in 0.0 0.5 0.9; do
    python3 round_length.py --intervals 500 --seeds 2 --budget 3000 --links wan:100:0.05 \
      --outer-lr "$lr" --outer-momentum "$mu" --out "results/sweep/lr${lr}_mu${mu}.json"
  done
done

# 3. Fixed-point diagnostic and heterogeneity runs (mild and strong shards, AdamW and memoryless SGD).
for het in mild strong; do
  for opt in adamw sgd; do
    python3 diag_experiment.py --hetero "$het" --opt "$opt" --out "results/diagnostic/${het}_${opt}.json"
  done
done
