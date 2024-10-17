import os
import pandas as pd
from torch.utils.data import Dataset
import logging
from PIL import Image, UnidentifiedImageError
import numpy as np


class AwA2Dataset(Dataset):
    def __init__(self, img_dir, attr_file, pred_file, transform=None):
        """
        Args:
        - img_dir (str): Directory with all the images.
        - attr_file (str): Path to the file with labels.
        - pred_file (str): Path to the file with symbolic tags.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.attributes = []
        self.symbolic_tags = []

        with open(attr_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                image_path = parts[0]
                label = int(parts[1])
                attribute = list(map(int, parts[2:])) if len(parts) > 2 else []
                self.image_paths.append(image_path)
                self.labels.append(label)
                self.attributes.append(attribute)

        # Load symbolic tags
        self.symbolic_tags = self.load_predicates(pred_file)

        # Ensure symbolic tags match the number of images
        self.match_symbolic_tags()

    def load_predicates(self, pred_file):
        """
        Load predicate matrix from file and ensure it is numeric.
        
        Args:
        - pred_file (str): Path to the predicate file.
        
        Returns:
        - np.ndarray: Predicate matrix as a numpy array of floats.
        """
        with open(pred_file, 'r') as file:
            lines = file.readlines()

        # Initialize a list to store valid rows
        data = []
        
        for line_num, line in enumerate(lines):
            parts = line.strip().split()

            try:
                # Convert each part to float; if it fails, skip this row
                numeric_parts = list(map(float, parts))
                data.append(numeric_parts)
            except ValueError as e:
                logging.error(f"Error converting line {line_num + 1} to float: {line.strip()}. Error: {e}")
                continue

        # Convert the cleaned data list into a DataFrame
        if len(data) == 0:
            raise ValueError("No valid rows found in the predicate file.")
        
        df = pd.DataFrame(data)

        return df.values


    def match_symbolic_tags(self):
        num_images = len(self.image_paths)
        num_tags = self.symbolic_tags.shape[0]

        if num_tags < num_images:
            # If there are fewer tags than images, pad with zeros
            pad_size = num_images - num_tags
            pad = np.zeros((pad_size, self.symbolic_tags.shape[1]))
            self.symbolic_tags = np.vstack((self.symbolic_tags, pad))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
        - idx (int): Index

        Returns:
        - tuple: (image, label, attribute, symbolic_tags)
        """
        image_path = os.path.join(self.img_dir, self.image_paths[idx])

        try:
            image = Image.open(image_path).convert('RGB')
        except UnidentifiedImageError:
            logging.error(f"Cannot identify image file {image_path}. It may be corrupted or not a valid image.")
            return None, None, None, None

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to tensor
        attribute = torch.tensor(self.attributes[idx], dtype=torch.float32)  # Convert attributes to tensor
        symbolic_tag = torch.tensor(self.symbolic_tags[idx], dtype=torch.float32)  # Convert symbolic_tag to tensor

        return image, label, attribute, symbolic_tag
