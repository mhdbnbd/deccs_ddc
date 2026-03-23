#!/bin/bash
set -e

RUNS=10

echo "=== DDECCS Full Experiment Suite ($RUNS runs each) ==="
echo ""

# AwA2 — 50 classes, 85 attributes
echo "===== AwA2 (K=50) ====="
for mode in kmeans deccs ddc ddeccs; do
    echo "--- AwA2 / $mode ---"
    python3 main_experiments.py --dataset awa2 --mode $mode --use_gpu --n_runs $RUNS
done

# aPY full — 32 classes, 64 attributes
echo "===== aPY full (K=32) ====="
for mode in kmeans deccs ddc ddeccs; do
    echo "--- aPY K=32 / $mode ---"
    python3 main_experiments.py --dataset apy --mode $mode --use_gpu --n_runs $RUNS
done

# aPY DDC subset — 15 classes (matching DDC paper exactly)
echo "===== aPY DDC-15 subset (K=15, matching DDC paper) ====="
for mode in kmeans deccs ddc ddeccs; do
    echo "--- aPY-15 / $mode ---"
    python3 main_experiments.py --dataset apy --mode $mode --use_gpu --apy_15 --n_runs $RUNS
done

echo ""
echo "=== All experiments complete ==="
echo "Results in: results/awa2/, results/apy/, results/apy_15/"