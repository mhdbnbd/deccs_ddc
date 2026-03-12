import argparse
import json
import logging
import os

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# disable nested parallelism to prevent spectral threadpool deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import time

import numpy as np
import torch
from numpy._typing import NDArray
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AwA2Dataset
from model import (
    Autoencoder, ConstrainedAutoencoder,
    ResNetAutoencoder, ResNetConstrainedAutoencoder,
    LargeAutoencoder, LargeConstrainedAutoencoder,
)
from train import train_autoencoder, train_constrained_autoencoder
from utils import (
    extract_embeddings,
    custom_collate,
    create_sample_dataset,
    setup_logging,
    save_detailed_results,
    evaluate_clustering,
    plot_experiment_results, get_base_clusterings, build_consensus_matrix, describe_clusters,
    summarize_clusters_with_attributes, generate_cluster_report, build_sparse_consensus, clustering_acc
)
from visualize_clusters import plot_tsne, save_cluster_examples, plot_deccs_loss


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

setup_logging()
device = select_device()
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description="AwA2 pipeline")
    parser.add_argument("--mode", type=str, choices=["ae", "oracle", "cae", "deccs"],
                        required=True, help="Experiment mode")
    parser.add_argument("--arch", type=str, choices=["small", "resnet", "large"],
                        default="small",
                        help="Encoder architecture: 'small' (original 4-layer CNN, 128x128), "
                             "'resnet' (pretrained ResNet-18, 224x224), "
                             "'large' (deeper CNN from scratch, 224x224)")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lambda_consensus", type=float, default=0.2)
    parser.add_argument("--tag_tuner", type=float, default=0.5,
                        help="Weight for tag supervision loss (CAE/DDC branch).")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save experiment results JSON (auto-generated if not set).")
    args = parser.parse_args()

    logging.info(f"=== Running mode: {args.mode.upper()} | arch: {args.arch.upper()} ===")

    # --- Build a unique prefix for all output files ---
    run_tag = f"{args.mode}_{args.arch}"
    if args.output_json is None:
        args.output_json = f"results_{run_tag}.json"

    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    dataset_dir = "AwA2-sample"
    pred_file = os.path.join(source_dir, "predicate-matrix-continuous.txt")
    classes_file = os.path.join(source_dir, "classes.txt")

    if args.use_sample:
        create_sample_dataset(source_dir, dataset_dir, classes_file, sample_size=1000)
        img_dir = os.path.join(dataset_dir, "JPEGImages")
        attr_file = os.path.join(dataset_dir, "AwA2-labels.txt")
    else:
        img_dir = os.path.join(source_dir, "JPEGImages")
        attr_file = os.path.join(source_dir, "AwA2-labels.txt")

    # --- Select transforms based on architecture ---
    if args.arch == "small":
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    else:  # resnet or large: both use 224x224
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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

    dataloader = DataLoader(
        awa2_dataset,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=custom_collate,
        shuffle=True)

    logging.info(f"Dataset created with {len(awa2_dataset)} samples.")
    logging.info(f"Architecture: {args.arch}")

    # --- Select model based on architecture and mode ---
    if args.mode in ["ae", "oracle"]:
        if args.arch == "small":
            model = Autoencoder().to(device)
        elif args.arch == "resnet":
            model = ResNetAutoencoder().to(device)
        else:  # large
            model = LargeAutoencoder().to(device)
        train_fn = train_autoencoder
    else:  # cae or deccs
        if args.arch == "small":
            model = ConstrainedAutoencoder().to(device)
        elif args.arch == "resnet":
            model = ResNetConstrainedAutoencoder().to(device)
        else:  # large
            model = LargeConstrainedAutoencoder().to(device)
        train_fn = train_constrained_autoencoder

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model params: {n_params:,} total, {n_trainable:,} trainable")

    start = time.time()
    training_losses = []
    loss_log = {"total": [], "recon": [], "tag": []}  # accumulate across epochs for npz
    for epoch in range(args.epochs):

        logging.info(f"Epoch {epoch + 1}/{args.epochs} started.")

        # --- Compute global consensus once per epoch (DECCS only) ---
        consensus_matrix = None
        if args.mode == "deccs":
            if (epoch == 0) or (epoch % 5 == 4): # build consensus_matrix every 5 epochs
                embeddings_np = extract_embeddings(dataloader, model, args.use_gpu).numpy()
                base_labels = get_base_clusterings(embeddings_np, n_clusters=len(np.unique(awa2_dataset.labels)))
                consensus_matrix = build_sparse_consensus(base_labels, embeddings_np)
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
            loss_log["total"].append(epoch_loss["total"])
            loss_log["recon"].append(epoch_loss["recon"])
            loss_log["tag"].append(epoch_loss["tag"])
            np.savez(f"results_{run_tag}_loss_components.npz",
                     total=np.array(loss_log["total"]),
                     recon=np.array(loss_log["recon"]),
                     tag=np.array(loss_log["tag"]))
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

    # --- Perform mode-specific clustering FIRST, then visualize with same labels ---
    if args.mode == "ae":
        results = evaluate_clustering(embeddings_np, true_labels)
    elif args.mode == "oracle":
        concat_features = np.concatenate([embeddings_np, symbolic_tags], axis=1)
        results = evaluate_clustering(concat_features, true_labels)
    elif args.mode == "deccs":
        base_labels = get_base_clusterings(embeddings_np, n_clusters=len(np.unique(true_labels)))
        # Use sparse consensus (consistent with training loop which uses build_sparse_consensus)
        consensus_matrix = build_sparse_consensus(base_labels, embeddings_np)

        final_labels = SpectralClustering(
            n_clusters=len(np.unique(true_labels)),
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42
        ).fit_predict(consensus_matrix)

        acc = clustering_acc(true_labels, final_labels)
        ari = adjusted_rand_score(true_labels, final_labels)
        nmi = normalized_mutual_info_score(true_labels, final_labels)
        sil = silhouette_score(embeddings_np, final_labels)
        results = {
            "acc": float(acc),
            "ari": float(ari),
            "nmi": float(nmi),
            "silhouette": float(sil),
            "clusters": final_labels.tolist()
        }
    else:  # cae
        results = evaluate_clustering(embeddings_np, true_labels)

    # --- Use the SAME cluster labels for visualization and descriptions ---
    cluster_labels = np.array(results["clusters"])

    # Compute attribute descriptions using the actual evaluation cluster labels
    # (not a separate KMeans run, which would be inconsistent)
    cluster_descriptions = []
    for c in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_descriptions.append(np.mean(symbolic_tags[mask], axis=0))
        else:
            cluster_descriptions.append(np.zeros(symbolic_tags.shape[1]))
    cluster_descriptions = np.array(cluster_descriptions)

    inspect_sample_clusters(awa2_dataset, cluster_labels)
    plot_tsne(embeddings.cpu().numpy(), cluster_labels, f"results_{run_tag}_tsne.png")
    samples_dir = save_cluster_examples(cluster_labels, awa2_dataset, mode=run_tag)

    summarize_clusters_with_attributes(
        cluster_labels=cluster_labels,
        cluster_descriptions=cluster_descriptions,
        dataset=awa2_dataset,
        predicates_path="data/AwA2-data/Animals_with_Attributes2/predicates.txt",
        top_k=5,
        output_json=f"results_{run_tag}_cluster_descriptions.json"
    )

    save_detailed_results(
        results={
            "mode": args.mode,
            "arch": args.arch,
            "training_losses": training_losses,
            "metrics": results,
        },
        output_path=args.output_json,
        lambda_consensus=args.lambda_consensus,
        tag_tuner=args.tag_tuner
    )
    logging.info(f"Experiment '{run_tag}' complete. Results saved to {args.output_json}")

    report = {
        "metrics": results,
        "arch": args.arch,
        "best_params": {
            "lambda_consensus": args.lambda_consensus,
            "tag_tuner": args.tag_tuner,
        },
        "model_params_total": n_params,
        "model_params_trainable": n_trainable,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"results_{run_tag}_summary.json", "w") as f:
        json.dump(report, f, indent=4)

    logging.info(f"Summary written to results_{run_tag}_summary.json")

    clusters = np.array(results["clusters"])
    # Normalize losses to flat list for plotting (CAE/DECCS return dicts)
    plot_losses = [l["total"] if isinstance(l, dict) else float(l) for l in training_losses]
    plot_experiment_results(
        output_dir=".",
        mode=run_tag,
        losses=plot_losses,
        embeddings=embeddings_np if args.mode != "oracle" else concat_features,
        clusters=clusters
    )
    generate_cluster_report(
        samples_dir=samples_dir,
        descriptions_json=f"results_{run_tag}_cluster_descriptions.json"
    )

    # Component-wise loss plot (only for modes that produce it)
    npz_path = f"results_{run_tag}_loss_components.npz"
    if args.mode in ["cae", "deccs"] and os.path.exists(npz_path):
        plot_deccs_loss(log_path=npz_path, save_path=f"results_{run_tag}_loss_components.png")
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