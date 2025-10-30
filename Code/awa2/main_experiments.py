import argparse
import logging
import os

import numpy as np
from numpy._typing import NDArray
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AwA2Dataset
from model import Autoencoder, ConstrainedAutoencoder
from train import train_autoencoder, train_constrained_autoencoder
from utils import (
    extract_embeddings,
    custom_collate,
    create_sample_dataset,
    setup_logging,
    save_detailed_results,
    evaluate_clustering,
    plot_experiment_results, get_base_clusterings, build_consensus_matrix
)

setup_logging()

def main():
    parser = argparse.ArgumentParser(description="AwA2 pipeline")
    parser.add_argument("--mode", type=str, choices=["ae", "oracle", "cae", "deccs"],
                        required=True, help="Experiment mode")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lambda_consensus", type=float, default=0.05)
    parser.add_argument("--k_clusters", type=int, default=10)
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

    labels_np = np.array(awa2_dataset.labels)
    tags_np = np.array(awa2_dataset.symbolic_tags)
    per_class_var = []
    for cls in np.unique(labels_np):
        v = tags_np[labels_np == cls].var(axis=0).mean()
        per_class_var.append(v)
    logging.info(f"Mean per-class tag variance: {np.mean(per_class_var):.6f}")

    dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    logging.info(f"Dataset created with {len(awa2_dataset)} samples.")

    if args.mode in ["ae", "oracle"]:
        model = Autoencoder()
        train_fn = train_autoencoder
    else:
        model = ConstrainedAutoencoder()
        train_fn = train_constrained_autoencoder

    training_losses = []
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs} started.")

        # --- Compute global consensus once per epoch (DECCS only) ---
        consensus_matrix = None
        if args.mode == "deccs":
            embeddings_np = extract_embeddings(dataloader, model, args.use_gpu).numpy()
            base_labels = get_base_clusterings(embeddings_np, n_clusters=args.k_clusters)
            consensus_matrix = build_consensus_matrix(base_labels)
            logging.info(f"[DECCS] Consensus matrix built for epoch {epoch + 1}")

        # --- Train epoch ---
        epoch_loss = train_fn(
            dataloader,
            model,
            args.use_gpu,
            consensus_matrix=consensus_matrix,  # <--- ADDED
            lambda_consensus=args.lambda_consensus if args.mode == "deccs" else 0.0
        )
        training_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.6f}")

    embeddings = extract_embeddings(dataloader, model, args.use_gpu)
    embeddings_np: NDArray[np.float32] = embeddings.detach().cpu().numpy().astype(np.float32)
    true_labels = np.array(awa2_dataset.labels)
    symbolic_tags = awa2_dataset.symbolic_tags

    if args.mode == "ae":
        results = evaluate_clustering(embeddings_np, true_labels)
    elif args.mode == "oracle":
        concat_features = np.concatenate([embeddings_np, symbolic_tags], axis=1)
        results = evaluate_clustering(concat_features, true_labels)
    else:  # cae / deccs
        results = evaluate_clustering(embeddings_np, true_labels)

    save_detailed_results(
        results={
            "mode": args.mode,
            "training_losses": training_losses,
            "metrics": results,
        },
        output_path=f"results_{args.mode}.json"
    )
    logging.info(f"Experiment '{args.mode}' complete. Results saved.")
    clusters = np.array(results["clusters"])
    plot_experiment_results(
        output_dir=".",
        mode=args.mode,
        losses=training_losses,
        embeddings=embeddings if args.mode != "oracle" else concat_features,
        clusters=clusters
    )

if __name__ == "__main__":
    main()