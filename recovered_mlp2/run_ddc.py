from model.ddc_autoencoder import ddc_Autoencoder
from utils import load_data, evaluate_clustering

def main():
    parser = argparse.ArgumentParser(description='Run DECCS with DDC integration.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--tags_path', type=str, required=True, help='Path to the tags dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    args = parser.parse_args()

    # Load data and tags
    data = load_data(args.data_path)
    tags = load_tags(args.tags_path)

    # Initialize and train the autoencoder
    autoencoder = ddc_Autoencoder(input_dim=data.shape[1])
    autoencoder.train(data, epochs=args.epochs, batch_size=args.batch_size)

    # Perform clustering
    clusters = autoencoder.cluster(data)

    # Generate explanations
    explanations = autoencoder.generate_explanations(clusters, data, tags)

    # Reconcile clustering with explanations
    pairwise_loss = autoencoder.reconcile_clustering_with_explanations(data, explanations)

    # Evaluate clustering results
    evaluate_clustering(data, clusters, explanations, pairwise_loss)

if __name__ == '__main__':
    main()
