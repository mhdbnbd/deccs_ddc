import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class AwA2Dataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        """
        Args:
        - img_dir (str): Directory with all the images.
        - attr_file (str): Path to the file with labels.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.attributes = []

        with open(attr_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                image_path = parts[0]
                label = int(parts[1])
                attribute = list(map(int, parts[2:])) if len(parts) > 2 else []
                self.image_paths.append(image_path)
                self.labels.append(label)
                self.attributes.append(attribute)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
        - idx (int): Index

        Returns:
        - tuple: (image, label, attribute) where attribute is optional
        """
        image_path = os.path.join(self.img_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        attribute = self.attributes[idx]
        return image, label, attribute
