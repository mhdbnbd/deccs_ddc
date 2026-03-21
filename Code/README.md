# DDECCS: Deep Descriptive Clustering with Consensus Representations

Project integrating [DDC](https://arxiv.org/abs/2105.11549) (Deep Descriptive Clustering) with [DECCS](https://arxiv.org/abs/2210.07063) (Deep Clustering with Consensus Representations) for interpretable image clustering.

## Overview

This project clusters images using pretrained ResNet-101 features and generates human-readable cluster explanations via Integer Linear Programming (ILP). It combines DECCS's consensus-based clustering robustness with DDC's interpretability framework.

**Pipeline:**
1. Extract frozen ResNet-101 features (2048-dim)
2. Cluster using K-means or DECCS consensus ensemble
3. Generate concise, orthogonal cluster descriptions via ILP
4. Evaluate with NMI, ACC, ARI, Silhouette, Tag Coverage, Inverse Tag Frequency

## Experiment Modes

| Mode | Clustering | Interpretability | Description |
|------|-----------|-----------------|-------------|
| `kmeans` | K-means | — | Baseline |
| `deccs` | DECCS consensus | — | Ensemble clustering |
| `ddc` | K-means | ILP descriptions | DDC interpretability |
| `ddeccs` | DECCS consensus | ILP descriptions | **Thesis contribution** |

## Datasets

| Dataset | Classes | Attributes | Images | Reference |
|---------|---------|-----------|--------|-----------|
| AwA2 | 50 | 85 | ~37K | [Xian et al., 2018](https://arxiv.org/abs/1707.00600) |
| aPY | 32 | 64 | ~15K | [Farhadi et al., 2009](https://vision.cs.uiuc.edu/attributes/) |

## Project Structure

```
Code/
├── main_experiments.py    # Main pipeline (feature extraction -> clustering -> ILP -> evaluation)
├── dataset.py             # Generic AttributeDataset loader (AwA2 + aPY)
├── visualize.py           # t-SNE and PCA visualizations
├── utils.py               # Consensus clustering, metrics, I/O utilities
├── setup_apy.py           # aPY dataset preprocessing script
├── run_all.sh             # Run all experiments (both datasets, all modes)
├── data/
│   ├── AwA2-data/         # Animals with Attributes 2
│   │   └── Animals_with_Attributes2/
│   │       ├── JPEGImages/
│   │       ├── classes.txt
│   │       ├── predicates.txt
│   │       ├── predicate-matrix-continuous.txt
│   │       └── labels.txt
│   └── aPY-data/          # Attribute Pascal & Yahoo
│       ├── attribute_data/    # Raw annotations
│       └── aPY/               # Preprocessed (created by setup_apy.py)
└── results/               # Output (git-ignored)
    ├── awa2/
    │   ├── kmeans/
    │   ├── deccs/
    │   ├── ddc/
    │   └── ddeccs/
    └── apy/
        └── ...
```

## Setup

### Requirements

```bash
pip install torch torchvision scikit-learn scipy numpy matplotlib tqdm pillow pulp
```

### Dataset Preparation

**AwA2** — download from [the official source](https://cvml.ista.ac.at/AwA2/):
```bash
# Place in data/AwA2-data/Animals_with_Attributes2/
# Ensure JPEGImages/, classes.txt, predicates.txt,
# predicate-matrix-continuous.txt, and labels.txt exist
```

**aPY** — requires Pascal VOC 2008 images + attribute annotations:
```bash
# 1. Download annotations
cd data/aPY-data
wget http://vision.cs.uiuc.edu/attributes/attribute_data.tar.gz
tar xzf attribute_data.tar.gz

# 2. Download Yahoo images (optional — Pascal images suffice for 20 classes)
wget http://vision.cs.uiuc.edu/attributes/ayahoo_test_images.tar.gz
tar xzf ayahoo_test_images.tar.gz

# 3. Preprocess into pipeline format
cd ../..
python3 setup_apy.py \
    --annotation_dir data/aPY-data/attribute_data \
    --pascal_images data/VOCdevkit/VOC2008/JPEGImages \
    --yahoo_images data/aPY-data/ayahoo_test_images \
    --output_dir data/aPY-data/aPY
```

## Usage

### Single experiment
```bash
# AwA2 with DDC interpretability
python3 main_experiments.py --dataset awa2 --mode ddc --use_gpu

# aPY with full DDECCS pipeline
python3 main_experiments.py --dataset apy --mode ddeccs --use_gpu

# With sampling (for quick testing)
python3 main_experiments.py --dataset awa2 --mode kmeans --use_sample --sample_size 2000
```

### Full experiment suite
```bash
bash run_all.sh
```

Results are saved to `results/<dataset>/<mode>/` containing:
- `summary.json` — metrics (NMI, ACC, ARI, Silhouette, TC, ITF)
- `ilp_descriptions.json` — ILP cluster descriptions (ddc/ddeccs modes)
- `cluster_descriptions.json` — attribute-based cluster summaries
- `tsne.png`, `pca.png` — visualizations
- `cluster_samples/` — representative images per cluster

## References

- Zhang & Davidson, "Deep Descriptive Clustering", IJCAI 2021
- Miklautz et al., "Deep Clustering with Consensus Representations", ICDM 2022
- Xian et al., "Zero-Shot Learning — A Comprehensive Evaluation of the Good, the Bad and the Ugly", TPAMI 2018
- Farhadi et al., "Describing Objects by their Attributes", CVPR 2009
