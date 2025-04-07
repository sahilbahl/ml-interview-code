import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans
from utils import generate_random_data


def main():
    """
    Example usage of KMeans implementation.
    """
    # Generate some random clustered data
    X, true_labels = generate_random_data(n_samples=300, n_features=2, n_centers=3)
    
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Get the predicted labels and centroids
    predicted_labels = kmeans.labels_
    centroids = kmeans.centroids
    
    # Plot the results
    plt.figure(figsize=(12, 5))
    
    # Plot ground truth
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
    plt.title('Ground Truth')
    
    # Plot KMeans result
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.title('KMeans Clustering')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Inertia: {kmeans.inertia_}")


if __name__ == "__main__":
    main()