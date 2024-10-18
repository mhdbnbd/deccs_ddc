#dataset.py

import os
import pandas as pd
from torch.utils.data import Dataset
import logging
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch


class AwA2Dataset(Dataset):
    def __init__(self, img_dir, attr_file, pred_file, transform=None):
        """
        Dataset for Animals with Attributes 2 (AwA2).
        Args:
        - img_dir (str): Directory containing images.
        - attr_file (str): Path to attribute file containing labels.
        - pred_file (str): Path to symbolic tag file.
        - transform (callable, optional): Transformations to apply on each image.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths, self.labels, self.attributes = self.load_attributes(attr_file)
        self.symbolic_tags = self.load_predicates(pred_file)
        
        # Ensures symbolic tags align with image count
        self.match_symbolic_tags()

    def load_attributes(self, attr_file):
        """
        Loads image paths, labels, and attributes from file.
        """
        image_paths, labels, attributes = [], [], []
        try:
            with open(attr_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    image_path = parts[0]
                    label = int(parts[1])
                    attribute = list(map(int, parts[2:])) if len(parts) > 2 else []
                    image_paths.append(image_path)
                    labels.append(label)
                    attributes.append(attribute)
        except Exception as e:
            logging.error(f"Error reading attributes file: {e}")
        return image_paths, labels, attributes

    def load_predicates(self, pred_file):
        """
        Loads predicates (symbolic tags) from file as numpy array.
        """
        data = []
        try:
            with open(pred_file, 'r') as file:
                for line_num, line in enumerate(file):
                    parts = line.strip().split()
                    try:
                        numeric_parts = list(map(float, parts))
                        data.append(numeric_parts)
                    except ValueError as e:
                        logging.error(f"Error parsing line {line_num + 1}: {e}")
        except Exception as e:
            logging.error(f"Error loading predicates file: {e}")

        if not data:
            raise ValueError("Predicate file contains no valid rows.")
        
        return np.array(data)

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
        """
        Retrieves image, label, attributes, and symbolic tags by index.
        """
        image_path = os.path.join(self.img_dir, self.image_paths[idx])

        try:
            image = Image.open(image_path).convert('RGB')
        except (UnidentifiedImageError, FileNotFoundError) as e:
            logging.error(f"Error loading image at {image_path}: {e}")
            return None, None, None, None

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        attribute = torch.tensor(self.attributes[idx], dtype=torch.float32)
        symbolic_tag = torch.tensor(self.symbolic_tags[idx], dtype=torch.float32)

        return image, label, attribute, symbolic_tag
