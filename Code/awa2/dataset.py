import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
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
        # Read the file and ensure the correct number of columns
        with open(pred_file, 'r') as file:
            lines = file.readlines()

        # Determine the number of columns (fields) expected
        num_columns = max(len(line.strip().split()) for line in lines)

        # Create a list to store the rows with the correct number of columns
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != num_columns:
                parts = parts[:num_columns]  # Trim extra fields
            data.append(parts)

        # Convert the list to a pandas DataFrame
        df = pd.DataFrame(data).astype(float)
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
        label = self.labels[idx]
        attribute = self.attributes[idx]
        symbolic_tag = self.symbolic_tags[idx]
        return image, label, attribute, symbolic_tag
