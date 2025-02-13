import os
import pandas as pd
from torch.utils.data import Dataset
import logging
from PIL import UnidentifiedImageError
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F

class AwA2Dataset(Dataset):
    def __init__(self, img_dir, attr_file, pred_file, classes_file, transform=None, train=True, train_ratio=0.8):
        """
        Args:
        - img_dir (str): Path to the directory containing images.
        - attr_file (str): Path to the attribute file (image-label mapping).
        - pred_file (str): Path to the predicate file (symbolic tag vectors for each class).
        - transform (callable, optional): Transformations to apply on images.
        - train (bool): Whether to use the training split or test split.
        - train_ratio (float): Ratio of dataset to use for training.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths, self.labels = self.load_image_labels(attr_file, classes_file)
        self.label_to_tags = self.load_predicates(pred_file)

        # Assign symbolic tags to each image based on its label
        self.symbolic_tags = np.array([self.label_to_tags[label] if label in self.label_to_tags else np.zeros(85) for label in self.labels])

        # Perform explicit train/test split
        split_index = int(len(self.image_paths) * train_ratio)
        if train:
            self.image_paths = self.image_paths[:split_index]
            self.labels = self.labels[:split_index]
            self.symbolic_tags = self.symbolic_tags[:split_index]
        else:
            self.image_paths = self.image_paths[split_index:]
            self.labels = self.labels[split_index:]
            self.symbolic_tags = self.symbolic_tags[split_index:]

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

            # Ensure mapping adjusts for 1-based to 0-based index shift
            return {i + 1: pred_matrix[i] for i in range(50)}  # Class ID 1 maps to index 0
        except Exception as e:
            logging.error(f"Error reading predicate file: {e}")
            return {}

    def match_symbolic_tags(self):
        """
        Ensures symbolic tags have the same number of rows as image paths.
        """
        num_images = len(self.image_paths)
        num_tags = self.symbolic_tags.shape[0]
        if num_tags != num_images:
            logging.warning(f"Number of images ({num_images}) does not match symbolic tags ({num_tags}). Padding with zeros.")
            pad_size = num_images - num_tags
            pad = np.zeros((pad_size, self.symbolic_tags.shape[1]))
            self.symbolic_tags = np.vstack([self.symbolic_tags, pad])

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
        print(f"Image: {img_path}, Expected Label: {self.labels[idx]}, "
              f"Tag Vector Shape: {symbolic_tag.shape}, First 5 Tags: {symbolic_tag[:5]}")

        return image, torch.tensor(symbolic_tag, dtype=torch.float32)

