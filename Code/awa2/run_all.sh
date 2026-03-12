#!/bin/bash
set -e

# Clean stale files from previous runs
rm -f results_deccs_loss_components.npz results_deccs_loss.png results_cluster_metrics.json

# === Small CNN (baseline) ===
python3 main_experiments.py --mode ae    --use_sample --epochs 30 --arch small
python3 main_experiments.py --mode cae   --use_sample --epochs 30 --arch small --tag_tuner 0.5
python3 main_experiments.py --mode deccs --use_sample --epochs 30 --arch small --lambda_consensus 0.2 --tag_tuner 0.5

# === Pretrained ResNet-18 ===
python3 main_experiments.py --mode ae    --use_sample --epochs 30 --arch resnet
python3 main_experiments.py --mode cae   --use_sample --epochs 30 --arch resnet --tag_tuner 0.5
python3 main_experiments.py --mode deccs --use_sample --epochs 30 --arch resnet --lambda_consensus 0.2 --tag_tuner 0.5

# === Large CNN ===
python3 main_experiments.py --mode ae    --use_sample --epochs 30 --arch large
python3 main_experiments.py --mode cae   --use_sample --epochs 30 --arch large --tag_tuner 0.5
python3 main_experiments.py --mode deccs --use_sample --epochs 30 --arch large --lambda_consensus 0.2 --tag_tuner 0.5

echo "=== All experiments complete ==="