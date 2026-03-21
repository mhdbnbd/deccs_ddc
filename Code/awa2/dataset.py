"""
Dataset loaders for attribute-based image datasets.

Supports:
  - AwA2: Animals with Attributes 2 (50 classes, 85 attributes, ~37K images)
  - aPY:  Attribute Pascal & Yahoo (32 classes, 64 attributes, ~15K images)

Both datasets share the same structure:
  - images organized in class-name folders
  - a predicate matrix mapping classes to attribute vectors
  - a classes.txt mapping class IDs to class names
"""

import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class AttributeDataset(Dataset):
    """
    Generic dataset for image classification with per-class semantic attributes.

    Works with any dataset that has:
      - Image directory with class-name subfolders
      - classes.txt: mapping class_id -> class_name
      - predicate-matrix-continuous.txt: (n_classes × n_attributes) matrix
      - labels file: image_path label pairs
    """

    def __init__(self, img_dir, attr_file, pred_file, classes_file,
                 transform=None, train=True, train_ratio=0.8):
        self.img_dir = img_dir
        self.transform = transform

        self.class_names, self.class_ids = self._load_classes(classes_file)
        self.image_paths, self.labels = self._load_image_labels(attr_file)
        self.label_to_tags = self._load_predicates(pred_file)

        n_attrs = next(iter(self.label_to_tags.values())).shape[0]
        self.symbolic_tags = np.array([
            self.label_to_tags.get(label, np.zeros(n_attrs))
            for label in self.labels
        ])

        missing = [l for l in set(self.labels) if l not in self.label_to_tags]
        if missing:
            logging.warning(f"Labels missing tags: {missing}")

        # Deterministic train/test split
        rng = np.random.default_rng(42)
        idx = np.arange(len(self.image_paths))
        rng.shuffle(idx)
        split = int(len(idx) * train_ratio)
        sel = idx[:split] if train else idx[split:]

        self.image_paths = [self.image_paths[i] for i in sel]
        self.labels = [self.labels[i] for i in sel]
        self.symbolic_tags = self.symbolic_tags[sel]

        logging.info(f"{'Train' if train else 'Test'} split: {len(self)} samples, "
                     f"{len(np.unique(self.labels))} classes, "
                     f"{self.symbolic_tags.shape[1]} attributes")

    def _load_classes(self, classes_file):
        names, ids = {}, {}
        with open(classes_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    cid = int(parts[0])
                    cname = parts[1]
                    names[cname] = cid
                    ids[cid] = cname
        return names, ids

    def _load_image_labels(self, attr_file):
        image_paths, labels = [], []
        with open(attr_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_rel = parts[0]
                class_folder = img_rel.split('/')[0]
                label = self.class_names.get(class_folder, -1)
                if label != -1:
                    image_paths.append(os.path.join(self.img_dir, img_rel))
                    labels.append(label)
        return image_paths, labels

    def _load_predicates(self, pred_file):
        pred_matrix = np.loadtxt(pred_file)
        n_classes, n_attrs = pred_matrix.shape
        logging.info(f"Predicate matrix: {n_classes} classes × {n_attrs} attributes")

        # Normalize to [0, 1]
        pmin, pmax = pred_matrix.min(), pred_matrix.max()
        if pmax > pmin:
            pred_matrix = (pred_matrix - pmin) / (pmax - pmin)

        # Map 1-based class IDs to predicate rows
        return {i + 1: pred_matrix[i] for i in range(n_classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        tag = self.symbolic_tags[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logging.warning(f"Error loading {img_path}: {e}")
            return None

        return image, torch.tensor(tag, dtype=torch.float32), idx


# Backward compatibility alias
AwA2Dataset = AttributeDataset


# =========================================================================
# Dataset configuration registry
# =========================================================================

DATASET_CONFIGS = {
    "awa2": {
        "name": "Animals with Attributes 2",
        "source_dir": "data/AwA2-data/Animals_with_Attributes2",
        "n_classes": 50,
        "n_attributes": 85,
    },
    "apy": {
        "name": "Attribute Pascal & Yahoo",
        "source_dir": "data/aPY-data/AttributePascalYahoo",
        "n_classes": 32,
        "n_attributes": 64,
    },
}


def get_dataset_paths(dataset_name, use_sample=False, sample_dir="samples"):
    """Get standardized paths for a dataset."""
    cfg = DATASET_CONFIGS[dataset_name]
    src = cfg["source_dir"]

    if use_sample:
        base = os.path.join(sample_dir, dataset_name)
        return {
            "img_dir": os.path.join(base, "JPEGImages"),
            "attr_file": os.path.join(base, "labels.txt"),
            "pred_file": os.path.join(src, "predicate-matrix-continuous.txt"),
            "classes_file": os.path.join(src, "classes.txt"),
            "predicates_file": os.path.join(src, "predicates.txt"),
            "sample_source": src,
        }
    else:
        return {
            "img_dir": os.path.join(src, "JPEGImages"),
            "attr_file": os.path.join(src, "labels.txt"),
            "pred_file": os.path.join(src, "predicate-matrix-continuous.txt"),
            "classes_file": os.path.join(src, "classes.txt"),
            "predicates_file": os.path.join(src, "predicates.txt"),
        }