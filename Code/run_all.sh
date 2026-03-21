#!/bin/bash
set -e

# Baseline: K-means on ResNet-101 features
python3 main_experiments.py --mode kmeans --use_gpu --n_clusters 50

# DECCS: consensus clustering on ResNet features
python3 main_experiments.py --mode deccs --use_gpu --n_clusters 50

# DDC: K-means + ILP interpretable descriptions
python3 main_experiments.py --mode ddc --use_gpu --n_clusters 50

# DDECCS: consensus + ILP descriptions (thesis contribution)
python3 main_experiments.py --mode ddeccs --use_gpu --n_clusters 50

echo "=== All experiments complete ==="