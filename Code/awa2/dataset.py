import logging
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class AwA2Dataset(Dataset):
    def __init__(self, img_dir, attr_file, pred_file, classes_file, transform=None, train=True, train_ratio=0.8):
        """
        Args:
        - img_dir (str): Path to the directory containing images.
        - attr_file (str): Path to the attribute file (image-label mapping).
        - pred_file (str): Path to the predicate file (symbolic tag vectors for each class).
        - classes_file: Path to classes.txt
        - transform (callable, optional): Transformations to apply on images.
        - train (bool): Whether to use the training split or test split.
        - train_ratio (float): Ratio of dataset to use for training.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.classes_file = classes_file
        self.image_paths, self.labels = self.load_image_labels(attr_file, classes_file)
        self.label_to_tags = self.load_predicates(pred_file)

        # Assign symbolic tags to each image based on its label
        self.symbolic_tags = np.array([self.label_to_tags[label] if label in self.label_to_tags else np.zeros(85) for label in self.labels])
        assert self.symbolic_tags.shape[1] == 85, f"Expected 85 tags, got {self.symbolic_tags.shape[1]}"
        missing_labels = [l for l in set(self.labels) if l not in self.label_to_tags]
        if missing_labels:
            logging.warning(f"Some labels missing tags: {missing_labels}")

        # Perform explicit train/test split
        rng = np.random.default_rng(42)
        idx = np.arange(len(self.image_paths))
        rng.shuffle(idx)
        split_index = int(len(idx) * train_ratio)
        train_idx, test_idx = idx[:split_index], idx[split_index:]
        if train:
            self.image_paths = [self.image_paths[i] for i in train_idx]
            self.labels = [self.labels[i] for i in train_idx]
            self.symbolic_tags = self.symbolic_tags[train_idx]
        else:
            self.image_paths = [self.image_paths[i] for i in test_idx]
            self.labels = [self.labels[i] for i in test_idx]
            self.symbolic_tags = self.symbolic_tags[test_idx]

    def load_image_labels(self, attr_file, classes_file):
        """
        Loads image file paths and their correct class labels from `classes.txt`.

        Args:
        - attr_file (str): Path to `AwA2-labels.txt`
        - classes_file (str): Path to `classes.txt`
        """
        image_paths, labels = [], []

        # Read correct class mapping from `classes.txt`
        class_mapping = {}
        with open(classes_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    class_id = int(parts[0])  # Keep `classes.txt` index
                    class_name = parts[1]
                    class_mapping[class_name] = class_id

        try:
            with open(attr_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    image_path = os.path.join(self.img_dir, parts[0])
                    class_folder = parts[0].split('/')[0]  # Extract class name from path
                    label = class_mapping.get(class_folder, -1)  # Use correct class ID

                    if label != -1:  # Ignore images with unknown labels
                        image_paths.append(image_path)
                        labels.append(label)
        except Exception as e:
            logging.error(f"Error reading attribute file: {e}")

        return image_paths, labels

    @staticmethod
    def load_predicates(pred_file):
        """Loads the predicate matrix and correctly maps 1-based class labels to 0-based predicate matrix indices."""
        try:
            pred_matrix = np.loadtxt(pred_file)  # Load all 85-dimensional class attributes

            if pred_matrix.shape != (50, 85):
                logging.warning(f"Expected (50, 85) shape, but got {pred_matrix.shape}")

            # Normalize values into [0,1] range
            pred_matrix = (pred_matrix - pred_matrix.min()) / (pred_matrix.max() - pred_matrix.min())
            symbolic_tags = {i + 1: pred_matrix[i] for i in range(pred_matrix.shape[0])}
            return symbolic_tags

        except Exception as e:
            logging.error(f"Error reading predicate file: {e}")
            return {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns an image and its corresponding 85-dimensional symbolic tag vector."""
        img_path = self.image_paths[idx]
        symbolic_tag = self.symbolic_tags[idx]  # Should have shape (85,)

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {e}")
            return None, None  # Skip bad images

        # Debugging the correct mapping
        logging.debug(f"Image: {img_path}, Expected Label: {self.labels[idx]}, "
              f"Tag Vector Shape: {symbolic_tag.shape}, First 5 Tags: {symbolic_tag[:5]}")

        return image, torch.tensor(symbolic_tag, dtype=torch.float32), idx

