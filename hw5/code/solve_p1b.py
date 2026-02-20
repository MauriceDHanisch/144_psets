import os
import numpy as np
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, SpectralClustering
from clustering_utils import plot_scatter_clusters, calculate_metrics

def solve_p1b():
    # Parameters
    n = 500
    noise = 0.01
    factor = 0.8 # Default as per Note 2
    k = 2
    
    # Ensure figs directory exists
    os.makedirs('latex/figs', exist_ok=True)
    
    # Generate Circle Graph
    X, true_labels = make_circles(n_samples=n, noise=noise, factor=factor, random_state=42)
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # 2. Spectral Clustering
    # For circle graph, we use affinity='nearest_neighbors' as suggested in Note 1
    spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
    spectral_labels = spectral.fit_predict(X)
    
    # Plotting
    plot_scatter_clusters(X, true_labels, f"Ground Truth Circle Graph (n={n}, noise={noise})", "latex/figs/p1b_ground_truth.png")
    plot_scatter_clusters(X, kmeans_labels, f"K-Means Clustering (ARI={calculate_metrics(true_labels, kmeans_labels):.3f})", "latex/figs/p1b_kmeans.png")
    plot_scatter_clusters(X, spectral_labels, f"Spectral Clustering (ARI={calculate_metrics(true_labels, spectral_labels):.3f})", "latex/figs/p1b_spectral.png")
    
    print(f"P1b: K-Means ARI: {calculate_metrics(true_labels, kmeans_labels):.3f}")
    print(f"P1b: Spectral ARI: {calculate_metrics(true_labels, spectral_labels):.3f}")

if __name__ == "__main__":
    solve_p1b()
