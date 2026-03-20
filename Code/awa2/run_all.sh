#!/bin/bash
set -e

rm -f results_deccs_loss_components.npz results_deccs_loss.png results_cluster_metrics.json

# === Small CNN baseline ===
python3 main_experiments.py --mode ae    --use_sample --epochs 50 --arch small
python3 main_experiments.py --mode cae   --use_sample --epochs 50 --arch small --tag_tuner 0.5
python3 main_experiments.py --mode deccs --use_sample --epochs 50 --arch small --lambda_consensus 0.2 --tag_tuner 0.5

# === DDCNet (thesis contribution) ===
#python3 main_experiments.py --mode ae    --use_sample --epochs 1   --arch resnet
python3 main_experiments.py --mode cae   --use_sample --epochs 500 --arch resnet --lambda_pairwise 3.0
python3 main_experiments.py --mode deccs --use_sample --epochs 500 --arch resnet --lambda_pairwise 3.0 --lambda_consensus 0.05

# === Large CNN ablation ===
python3 main_experiments.py --mode ae    --use_sample --epochs 50 --arch large
python3 main_experiments.py --mode cae   --use_sample --epochs 50 --arch large --tag_tuner 0.5
python3 main_experiments.py --mode deccs --use_sample --epochs 50 --arch large --lambda_consensus 0.2 --tag_tuner 0.5

echo "=== All experiments complete ==="
