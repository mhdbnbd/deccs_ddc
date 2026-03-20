import argparse
import json
import logging
import os

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

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
    DDCNet,
    LargeAutoencoder, LargeConstrainedAutoencoder,
)
from train import train_autoencoder, train_constrained_autoencoder, train_ddc
from utils import (
    extract_embeddings,
    custom_collate,
    create_sample_dataset,
    setup_logging,
    save_detailed_results,
    evaluate_clustering,
    plot_experiment_results, get_base_clusterings,
    summarize_clusters_with_attributes, generate_cluster_report,
    build_sparse_consensus, clustering_acc
)
from visualize_clusters import plot_tsne, save_cluster_examples, plot_deccs_loss


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
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
    parser = argparse.ArgumentParser(description="AwA2 DECCS-DDC pipeline")
    parser.add_argument("--mode", type=str, choices=["ae", "oracle", "cae", "deccs"],
                        required=True, help="Experiment mode")
    parser.add_argument("--arch", type=str, choices=["small", "resnet", "large"],
                        default="small",
                        help="'small' (4-layer CNN 128x128), "
                             "'resnet' (DDCNet: frozen ResNet-101 + MLP), "
                             "'large' (deeper CNN 224x224)")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lambda_consensus", type=float, default=0.2)
    parser.add_argument("--lambda_pairwise", type=float, default=1.0,
                        help="Weight for DDC pairwise constraint loss (resnet arch only).")
    parser.add_argument("--lambda_tag", type=float, default=1.0,
                        help="Weight for tag BCE prediction loss (resnet arch only).")
    parser.add_argument("--tag_tuner", type=float, default=0.5,
                        help="Weight for tag supervision loss (small/large arch).")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=2000,
                        help="Number of images to sample (only with --use_sample).")
    args = parser.parse_args()

    run_tag = f"{args.mode}_{args.arch}"
    if args.output_json is None:
        args.output_json = f"results_{run_tag}.json"

    logging.info(f"=== Running mode: {args.mode.upper()} | arch: {args.arch.upper()} ===")

    # --- Dataset ---
    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    dataset_dir = "AwA2-sample"
    pred_file = os.path.join(source_dir, "predicate-matrix-continuous.txt")
    classes_file = os.path.join(source_dir, "classes.txt")

    if args.use_sample:
        create_sample_dataset(source_dir, dataset_dir, classes_file, sample_size=args.sample_size)
        img_dir = os.path.join(dataset_dir, "JPEGImages")
        attr_file = os.path.join(dataset_dir, "AwA2-labels.txt")
    else:
        img_dir = os.path.join(source_dir, "JPEGImages")
        attr_file = os.path.join(source_dir, "AwA2-labels.txt")

    # --- Transforms ---
    if args.arch == "small":
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    elif args.arch == "resnet":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    awa2_dataset = AwA2Dataset(
        img_dir=img_dir, attr_file=attr_file, pred_file=pred_file,
        classes_file=classes_file, transform=transform, train=True
    )

    n_classes = len(np.unique(awa2_dataset.labels))
    logging.info(f"Dataset: {len(awa2_dataset)} samples, {n_classes} classes")

    dataloader = DataLoader(
        awa2_dataset, batch_size=64, num_workers=4,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2, collate_fn=custom_collate, shuffle=True
    )

    # --- Model selection ---
    is_ddc = (args.arch == "resnet" and args.mode in ["cae", "deccs"])
    skip_training = (args.arch == "resnet" and args.mode in ["ae", "oracle"])

    if args.arch == "resnet":
        model = DDCNet(n_clusters=n_classes).to(device)
        train_fn = train_ddc
    elif args.mode in ["ae", "oracle"]:
        model = (Autoencoder() if args.arch == "small" else LargeAutoencoder()).to(device)
        train_fn = train_autoencoder
    else:
        model = (ConstrainedAutoencoder() if args.arch == "small"
                 else LargeConstrainedAutoencoder()).to(device)
        train_fn = train_constrained_autoencoder

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Architecture: {args.arch} | Params: {n_params:,} total, {n_trainable:,} trainable")

    # --- Training loop ---
    start = time.time()
    training_losses = []
    loss_log = {"total": [], "recon": [], "tag": []}

    if skip_training:
        logging.info("Skipping training (frozen ResNet encoder, AE/oracle mode).")
    else:
        cached_features = None  # Will be populated on first DDC epoch
        for epoch in range(args.epochs):
            logging.info(f"Epoch {epoch + 1}/{args.epochs} started.")

            # Build consensus for DECCS mode (every 5 epochs)
            consensus_matrix = None
            if args.mode == "deccs":
                if (epoch == 0) or (epoch % 5 == 4):
                    emb = extract_embeddings(dataloader, model, args.use_gpu).numpy()
                    base_labels = get_base_clusterings(emb, n_clusters=n_classes)
                    consensus_matrix = build_sparse_consensus(base_labels, emb)
                    logging.info(f"[DECCS] Consensus matrix built for epoch {epoch + 1}")

            # Train one epoch
            if is_ddc:
                epoch_loss = train_fn(
                    dataloader, model, args.use_gpu,
                    lambda_pairwise=args.lambda_pairwise,
                    lambda_tag=args.lambda_tag,
                    consensus_matrix=consensus_matrix,
                    lambda_consensus=args.lambda_consensus if args.mode == "deccs" else 0.0,
                    _cached_features=cached_features,
                )
                # Cache extracted features (only computed once)
                if isinstance(epoch_loss, dict) and "_cached_features" in epoch_loss:
                    cached_features = epoch_loss.pop("_cached_features")
            else:
                epoch_loss = train_fn(
                    dataloader, model, args.use_gpu,
                    consensus_matrix=consensus_matrix,
                    lambda_consensus=args.lambda_consensus if args.mode == "deccs" else 0.0,
                    tag_tuner=args.tag_tuner,
                )

            training_losses.append(epoch_loss)
            if isinstance(epoch_loss, dict):
                for k in ["total", "recon", "tag"]:
                    loss_log[k].append(epoch_loss.get(k, 0.0))
                np.savez(f"results_{run_tag}_loss_components.npz",
                         total=np.array(loss_log["total"]),
                         recon=np.array(loss_log["recon"]),
                         tag=np.array(loss_log["tag"]))
                logging.info(f"Epoch {epoch + 1}/{args.epochs} completed. "
                             f"Total={epoch_loss['total']:.4f}")
            else:
                logging.info(f"Epoch {epoch + 1} completed. Loss: {float(epoch_loss):.6f}")

            logging.info(f"Epoch time: {time.time() - start:.2f}s")

    # --- Extract final embeddings ---
    if is_ddc and cached_features is not None:
        # Reuse cached ResNet features — avoids re-running backbone (hours on CPU)
        logging.info("Using cached features for evaluation (skipping ResNet re-extraction).")
        cached_feat, cached_tags, cached_idx = cached_features
        cached_feat = cached_feat.to(device)
        model.eval()
        with torch.no_grad():
            out = model.forward_from_features(cached_feat)
            embeddings_np = out['embeddings'].cpu().numpy().astype(np.float32)
            final_labels = out['cluster_probs'].argmax(dim=1).cpu().numpy()
        embeddings = torch.tensor(embeddings_np)
    else:
        embeddings = extract_embeddings(dataloader, model, args.use_gpu)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
        final_labels = None

    true_labels = np.array(awa2_dataset.labels)
    symbolic_tags = awa2_dataset.symbolic_tags

    # --- Clustering & Evaluation ---
    if is_ddc or skip_training:
        if final_labels is None:
            # Fallback: run through dataloader (slow for large datasets)
            logging.info("Using DDCNet's direct cluster assignments for evaluation.")
            model.eval()
            all_assignments = []
            with torch.no_grad():
                for batch in dataloader:
                    if batch is None:
                        continue
                    images = batch[0].to(device)
                    assignments = model.get_cluster_assignments(images)
                    all_assignments.append(assignments.cpu().numpy())
            final_labels = np.concatenate(all_assignments)

        acc = clustering_acc(true_labels, final_labels)
        ari = adjusted_rand_score(true_labels, final_labels)
        nmi = normalized_mutual_info_score(true_labels, final_labels)
        sil = silhouette_score(embeddings_np, final_labels) if len(np.unique(final_labels)) > 1 else 0.0
        results = {
            "acc": float(acc), "ari": float(ari), "nmi": float(nmi),
            "silhouette": float(sil), "clusters": final_labels.tolist()
        }
        logging.info(f"DDCNet ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}, Sil={sil:.4f}")

    elif args.mode == "deccs":
        # Non-DDC DECCS: consensus clustering on embeddings
        base_labels = get_base_clusterings(embeddings_np, n_clusters=n_classes)
        consensus_matrix = build_sparse_consensus(base_labels, embeddings_np)
        final_labels = SpectralClustering(
            n_clusters=n_classes, affinity="precomputed",
            assign_labels="kmeans", random_state=42
        ).fit_predict(consensus_matrix)

        acc = clustering_acc(true_labels, final_labels)
        ari = adjusted_rand_score(true_labels, final_labels)
        nmi = normalized_mutual_info_score(true_labels, final_labels)
        sil = silhouette_score(embeddings_np, final_labels)
        results = {
            "acc": float(acc), "ari": float(ari), "nmi": float(nmi),
            "silhouette": float(sil), "clusters": final_labels.tolist()
        }
        logging.info(f"Consensus ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}, Sil={sil:.4f}")

    elif args.mode == "oracle":
        concat = np.concatenate([embeddings_np, symbolic_tags], axis=1)
        results = evaluate_clustering(concat, true_labels)
    else:
        results = evaluate_clustering(embeddings_np, true_labels)

    # --- Visualization & Saving ---
    cluster_labels = np.array(results["clusters"])

    # Compute attribute descriptions from the actual cluster labels
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
        cluster_labels=cluster_labels, cluster_descriptions=cluster_descriptions,
        dataset=awa2_dataset,
        predicates_path="data/AwA2-data/Animals_with_Attributes2/predicates.txt",
        top_k=5, output_json=f"results_{run_tag}_cluster_descriptions.json"
    )

    save_detailed_results(
        results={"mode": args.mode, "arch": args.arch,
                 "training_losses": training_losses, "metrics": results},
        output_path=args.output_json,
        lambda_consensus=args.lambda_consensus, tag_tuner=args.tag_tuner
    )

    report = {
        "metrics": {k: v for k, v in results.items() if k != "clusters"},
        "arch": args.arch,
        "params": {"lambda_consensus": args.lambda_consensus,
                   "lambda_pairwise": args.lambda_pairwise,
                   "tag_tuner": args.tag_tuner},
        "model_params_total": n_params,
        "model_params_trainable": n_trainable,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"results_{run_tag}_summary.json", "w") as f:
        json.dump(report, f, indent=4)
    logging.info(f"Summary written to results_{run_tag}_summary.json")

    # Plots
    plot_losses = [l["total"] if isinstance(l, dict) else float(l) for l in training_losses]
    plot_experiment_results(
        output_dir=".", mode=run_tag, losses=plot_losses,
        embeddings=embeddings_np, clusters=cluster_labels
    )
    generate_cluster_report(
        samples_dir=samples_dir,
        descriptions_json=f"results_{run_tag}_cluster_descriptions.json"
    )
    npz_path = f"results_{run_tag}_loss_components.npz"
    if os.path.exists(npz_path):
        plot_deccs_loss(log_path=npz_path, save_path=f"results_{run_tag}_loss_components.png")

    logging.info(f"Total time: {time.time() - start:.2f}s")


def inspect_sample_clusters(dataset, cluster_labels, num_clusters=3, samples_per_cluster=3):
    selected = random.sample(list(set(cluster_labels)), min(num_clusters, len(set(cluster_labels))))
    for cluster in selected:
        indices = np.where(cluster_labels == cluster)[0]
        chosen = random.sample(list(indices), min(samples_per_cluster, len(indices)))
        print(f"\nCluster {cluster}: showing {len(chosen)} samples")
        for idx in chosen:
            print(f"  - Image: {dataset.image_paths[idx]} | Label: {dataset.labels[idx]}")


if __name__ == "__main__":
    main()