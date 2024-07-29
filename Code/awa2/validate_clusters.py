import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, silhouette_score
from collections import Counter

# Path to the JSON file containing clustering results
CLUSTER_RESULTS_FILE = 'clustering_results.json'

# Function to load clustering results from a JSON file
def load_clusters_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Extract clusters and sort them by image index
    clusters = np.array([data[f"Image_{i}"] for i in range(len(data))])
    return clusters

# Step 1: Inspect Clusters Distribution
def print_cluster_distribution(clusters):
    cluster_counts = Counter(clusters)
    print("Cluster distribution:", cluster_counts)

# Step 2: Visualize Clusters
def plot_clusters(embeddings, clusters):
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.title("Cluster Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.colorbar(label='Cluster')
    plt.show()

# Step 3: Cluster Centroids Analysis
def analyze_centroids(embeddings, clusters):
    kmeans = KMeans(n_clusters=len(np.unique(clusters)))
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    print("Cluster Centroids:\n", centroids)

# Step 4: Check Homogeneity of Clusters
def compute_homogeneity(true_labels, clusters):
    homogeneity = homogeneity_score(true_labels, clusters)
    print("Homogeneity Score:", homogeneity)

# Step 5: Inspect Cluster Content
def inspect_clusters(data, clusters, n_samples=5):
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        print(f"\nCluster {cluster}:")
        cluster_indices = np.where(clusters == cluster)[0]
        sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        for idx in sample_indices:
            print(data[idx])

# Step 6: Evaluate Clustering Performance
def compute_silhouette_score(embeddings, clusters):
    silhouette_avg = silhouette_score(embeddings, clusters)
    print("Silhouette Score:", silhouette_avg)

# Run all validations
def main():
    clusters = load_clusters_from_json(CLUSTER_RESULTS_FILE)

    # Replace these with actual data
    embeddings = np.random.rand(100, 2)  # Example 2D embeddings, replace with your actual embeddings
    true_labels = np.random.randint(0, 5, size=100)  # Example true labels, replace with your actual labels

    print_cluster_distribution(clusters)
    plot_clusters(embeddings, clusters)
    analyze_centroids(embeddings, clusters)
    compute_homogeneity(true_labels, clusters)
    data = np.random.rand(100, 10)  # Example data, replace with actual data
    inspect_clusters(data, clusters)
    compute_silhouette_score(embeddings, clusters)

if __name__ == "__main__":
    main()

#python3 validate_clusters.py
