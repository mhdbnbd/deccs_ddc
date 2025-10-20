import argparse
import logging
import os

import numpy as np
from dataset import AwA2Dataset
from model import Autoencoder, ConstrainedAutoencoder
from torch.utils.data import DataLoader
from torchvision import transforms
from train import train_autoencoder, train_constrained_autoencoder, evaluate_autoencoder
from utils import (
    extract_embeddings,
    custom_collate,
    create_sample_dataset,
    setup_logging,
    save_detailed_results,
    clustering_acc,
    evaluate_clustering
)

setup_logging()

def main():
    parser = argparse.ArgumentParser(description="AwA2 pipeline")
    parser.add_argument("--mode", type=str, choices=["baseline", "concat", "cae"],
                        required=True, help="Experiment mode")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    logging.info(f"=== Running mode: {args.mode.upper()} ===")

    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    dataset_dir = "AwA2-sample"
    pred_file = os.path.join(source_dir, "predicate-matrix-continuous.txt")
    classes_file = os.path.join(source_dir, "classes.txt")

    if args.use_sample:
        create_sample_dataset(source_dir, dataset_dir, classes_file, sample_size=100)
        img_dir = os.path.join(dataset_dir, "JPEGImages")
        attr_file = os.path.join(dataset_dir, "AwA2-labels.txt")
    else:
        img_dir = os.path.join(source_dir, "JPEGImages")
        attr_file = os.path.join(source_dir, "AwA2-labels.txt")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    awa2_dataset = AwA2Dataset(
        img_dir=img_dir,
        attr_file=attr_file,
        pred_file=pred_file,
        classes_file=classes_file,
        transform=transform,
        train=True
    )
    dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    logging.info(f"Dataset created with {len(awa2_dataset)} samples.")

    if args.mode == "baseline" or args.mode == "concat":
        model = Autoencoder()
        train_fn = train_autoencoder
    else:
        model = ConstrainedAutoencoder()
        train_fn = train_constrained_autoencoder

    training_losses = []
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs} started.")
        epoch_loss = train_fn(dataloader, model, args.use_gpu)
        training_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.6f}")

    embeddings = extract_embeddings(dataloader, model, args.use_gpu)
    true_labels = np.array(awa2_dataset.labels)
    symbolic_tags = awa2_dataset.symbolic_tags

    if args.mode == "baseline":
        results = evaluate_clustering(embeddings, true_labels, mode_desc="Baseline")
    elif args.mode == "concat":
        concat_features = np.concatenate([embeddings, symbolic_tags], axis=1)
        results = evaluate_clustering(concat_features, true_labels, mode_desc="Concat (tags post-training)")
    else:
        results = evaluate_clustering(embeddings, true_labels, mode_desc="Constrained AE (tags supervised)")
    save_detailed_results(
        results={
            "mode": args.mode,
            "training_losses": training_losses,
            "metrics": results,
        },
        output_path=f"results_{args.mode}.json"
    )
    logging.info(f"Experiment '{args.mode}' complete. Results saved.")

if __name__ == "__main__":
    main()
