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

        # Load the image using PIL
        try:
            image = Image.open(image_path)

            # Ensure the image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            logging.debug(f"Loaded image from {image_path}")

        except UnidentifiedImageError as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None, None, None, None

        # Apply transformations if they exist, before converting to a Tensor
        if self.transform:
            try:
                # Apply transformations to the image (PIL or ndarray)
                image = self.transform(image)
                logging.debug(f"Image transformed successfully for {image_path}")

            except Exception as e:
                logging.error(f"Failed to apply transformations to image {image_path}: {e}")
                return None, None, None, None

        # Convert to Tensor if not already a Tensor
        if isinstance(image, torch.Tensor):
            image_tensor = image  # If already a tensor, assign it directly.
        else:
            # Ensure the image is converted to NumPy array if still in PIL/ndarray form
            try:
                image_np = np.array(image, dtype=np.uint8)
                logging.debug(f"Converted image to NumPy array: {image_np.shape}")

                # Convert NumPy array to PyTorch tensor (Channel, Height, Width)
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # Rescale to [0, 1]
                logging.debug(f"Converted image to tensor: {image_tensor.shape}")

            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")
                return None, None, None, None

        # label = torch.tensor(self.labels[idx], dtype=torch.long)
        # attribute = torch.tensor(self.attributes[idx], dtype=torch.float32)
        symbolic_tag = torch.tensor(self.symbolic_tags[idx], dtype=torch.float32)

        return image_tensor, symbolic_tag

