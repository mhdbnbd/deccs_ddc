#!/bin/bash
set -e

echo "=== DDECCS Experiment Suite ==="

# AwA2 (50 classes, 85 attributes)
for mode in kmeans deccs ddc ddeccs; do
    echo "--- AwA2 / $mode ---"
    python3 main_experiments.py --dataset awa2 --mode $mode --use_gpu
done

# aPY (32 classes, 64 attributes)
for mode in kmeans deccs ddc ddeccs; do
    echo "--- aPY / $mode ---"
    python3 main_experiments.py --dataset apy --mode $mode --use_gpu
done

echo "=== All experiments complete ==="
echo "Results in: results/awa2/ and results/apy/"