#utils.py

import torch
import os
import logging
import shutil
import random
import logging

def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_embeddings(dataloader, model, use_gpu):
    """
    Extract embeddings using the trained autoencoder model.
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch_idx, (images, _, _, _) in enumerate(dataloader):
            images = images.to(device)
            logging.debug(f"Batch {batch_idx} processed with image shape: {images.shape}")

            # Forward pass through autoencoder encoder
            encoded = model.encoder(images)
            logging.debug(f"Encoded embeddings for batch {batch_idx}: {encoded.shape}")

            embeddings.append(encoded.view(encoded.size(0), -1).cpu())

    embeddings = torch.cat(embeddings, dim=0)
    logging.info(f"Total extracted embeddings shape: {embeddings.shape}")
    return embeddings

def create_sample_dataset(source_dir, target_dir, sample_size=100):
    """
    Create a sample dataset from the source dataset.
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    img_dir = os.path.join(source_dir, "JPEGImages")
    labels_file = os.path.join(source_dir, "AwA2-labels.txt")
    sample_img_dir = os.path.join(target_dir, "JPEGImages")
    sample_labels_file = os.path.join(target_dir, "AwA2-labels.txt")

    if not os.path.exists(sample_img_dir):
        os.makedirs(sample_img_dir, exist_ok=True)

    if not os.path.exists(labels_file):
        logging.warning(f"Labels file {labels_file} not found. Generating a new one.")
        generate_labels_file(img_dir, labels_file)

    all_images = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith('.jpg'):
                all_images.append(os.path.join(root, file))

    if sample_size > len(all_images):
        sample_size = len(all_images)
        logging.warning(f"Sample size adjusted to {sample_size} due to limited number of images.")

    sampled_images = random.sample(all_images, sample_size)

    for img in sampled_images:
        target_path = os.path.join(sample_img_dir, os.path.relpath(img, img_dir))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if os.path.exists(target_path):
            os.remove(target_path)

        shutil.copy(img, target_path)
        logging.debug(f"Copied {img} to {target_path}")

    with open(labels_file, "r") as f:
        lines = f.readlines()

    sampled_lines = [line for line in lines if os.path.relpath(line.split()[0], '').replace('\\', '/') in [os.path.relpath(img, img_dir).replace('\\', '/') for img in sampled_images]]

    with open(sample_labels_file, "w") as f:
        f.writelines(sampled_lines)

    logging.info(f"Sample of {sample_size} images created in {sample_img_dir}")
    logging.info(f"Sample labels file created at {sample_labels_file}")

def generate_labels_file(img_dir, labels_file):
    """
    Generate a labels file for the dataset.

    Parameters:
    img_dir (str): Directory containing the image folders.
    labels_file (str): Path to the labels file to be created.
    """
    # Ensure the directory for the labels file exists
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)

    with open(labels_file, 'w') as f:
        for class_label, class_dir in enumerate(os.listdir(img_dir)):
            class_path = os.path.join(img_dir, class_dir)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    if img_filename.endswith('.jpg'):
                        f.write(f"{os.path.join(class_dir, img_filename)} {class_label}\n")
    logging.info(f"Labels file created at {labels_file}")

def custom_collate(batch):
    # Filter out None samples
    batch = [b for b in batch if b[0] is not None]
    
    if len(batch) == 0:
        logging.warning("All samples in the batch are None. Skipping this batch.")
        raise StopIteration  # This prevents the training loop from crashing

    return torch.utils.data.default_collate(batch)