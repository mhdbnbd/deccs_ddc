import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from numpy._typing import NDArray
from torch.utils.data import DataLoader
from torchvision import transforms

from visualize_clusters import plot_tsne, save_cluster_examples
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
    plot_experiment_results, get_base_clusterings, build_consensus_matrix, describe_clusters,
    summarize_clusters_with_attributes, generate_cluster_report
)


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        logging.info(f"Using GPU: {gpu_name} ({total_mem:.2f} GB total)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device

device = select_device()

setup_logging()

def main():
    parser = argparse.ArgumentParser(description="AwA2 pipeline")
    parser.add_argument("--mode", type=str, choices=["ae", "oracle", "cae", "deccs"],
                        required=True, help="Experiment mode")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lambda_consensus", type=float, default=0.2)
    parser.add_argument("--tag_tuner", type=float, default=0.5,
                        help="Weight for tag supervision loss (CAE/DDC branch).")
    parser.add_argument("--output_json", type=str, default="results_deccs.json",
                        help="Path to save experiment results JSON.")
    args = parser.parse_args()

    logging.info(f"=== Running mode: {args.mode.upper()} ===")

    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    dataset_dir = "AwA2-sample"
    pred_file = os.path.join(source_dir, "predicate-matrix-continuous.txt")
    classes_file = os.path.join(source_dir, "classes.txt")

    if args.use_sample:
        create_sample_dataset(source_dir, dataset_dir, classes_file, sample_size=200)
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
        model = Autoencoder().to(device)
        train_fn = train_autoencoder
    else:
        model = ConstrainedAutoencoder().to(device)
        train_fn = train_constrained_autoencoder

    start = time.time()
    training_losses = []
    for epoch in range(args.epochs):

        logging.info(f"Epoch {epoch + 1}/{args.epochs} started.")

        # --- Compute global consensus once per epoch (DECCS only) ---
        consensus_matrix = None
        if args.mode == "deccs":
            embeddings_np = extract_embeddings(dataloader, model, args.use_gpu).numpy()
            base_labels = get_base_clusterings(embeddings_np, n_clusters=len(np.unique(awa2_dataset.labels)))
            consensus_matrix = build_consensus_matrix(base_labels)
            logging.info(f"[DECCS] Consensus matrix built for epoch {epoch + 1}")

        # --- Train epoch ---
        epoch_loss = train_fn(
            dataloader,
            model,
            args.use_gpu,
            consensus_matrix=consensus_matrix,
            lambda_consensus=args.lambda_consensus if args.mode == "deccs" else 0.0,
            tag_tuner=args.tag_tuner
        )
        training_losses.append(epoch_loss)
        if isinstance(epoch_loss, dict):
            logging.info(
                f"Epoch {epoch + 1}/{args.epochs} completed. "
                f"Recon={epoch_loss['recon']:.4f}, "
                f"Tag={epoch_loss['tag']:.4f}, "
                f"Total={epoch_loss['total']:.4f}"
            )
        else:
            loss_val = float(np.mean(epoch_loss)) if isinstance(epoch_loss, (list, tuple)) else float(epoch_loss)
            logging.info(f"Epoch {epoch + 1} completed. Loss: {loss_val:.6f}")

        logging.info(f"Epoch time: {time.time() - start:.2f}s")

    embeddings = extract_embeddings(dataloader, model, args.use_gpu)
    embeddings_np: NDArray[np.float32] = embeddings.detach().cpu().numpy().astype(np.float32)
    true_labels = np.array(awa2_dataset.labels)
    symbolic_tags = awa2_dataset.symbolic_tags
    cluster_labels, cluster_descriptions = describe_clusters(embeddings, symbolic_tags)
    inspect_sample_clusters(awa2_dataset, cluster_labels)
    plot_tsne(embeddings.cpu().numpy(), cluster_labels, "results_tsne.png")
    save_cluster_examples(cluster_labels, awa2_dataset)

    summarize_clusters_with_attributes(
        cluster_labels=cluster_labels,
        cluster_descriptions=cluster_descriptions,
        dataset=awa2_dataset,
        predicates_path="data/AwA2-data/Animals_with_Attributes2/predicates.txt",
        top_k=5,
        output_json="results_cluster_descriptions.json"
    )

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
        output_path=args.output_json
    )
    logging.info(f"Experiment '{args.mode}' complete. Results saved.")

    report = {
        "metrics": results,
        "best_params": {
            "lambda_consensus": args.lambda_consensus,
            "tag_tuner": args.tag_tuner,
        },
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("results_summary.json", "w") as f:
        json.dump(report, f, indent=4)

    logging.info("Summary written to results_summary.json")

    clusters = np.array(results["clusters"])
    plot_experiment_results(
        output_dir=".",
        mode=args.mode,
        losses=training_losses,
        embeddings=embeddings if args.mode != "oracle" else concat_features,
        clusters=clusters
    )
    generate_cluster_report()
    logging.info(f"Total time: {time.time() - start:.2f}s")

def inspect_sample_clusters(dataset, cluster_labels, num_clusters=3, samples_per_cluster=3):
    """
    Print sample image paths from selected clusters for visual verification.
    """
    selected_clusters = random.sample(list(set(cluster_labels)), num_clusters)
    for cluster in selected_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        chosen = random.sample(list(indices), min(samples_per_cluster, len(indices)))
        print(f"\nCluster {cluster}: showing {len(chosen)} samples")
        for idx in chosen:
            img_path = dataset.image_paths[idx]
            label = dataset.labels[idx]
            print(f"  - Image: {img_path} | Label: {label}")

if __name__ == "__main__":
    main()