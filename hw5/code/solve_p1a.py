import os
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from clustering_utils import generate_ssbm, plot_graph_clusters, calculate_metrics

def solve_p1a():
    # Parameters
    n = 30
    k = 3
    A = 0.7
    B = 0.1
    
    # Ensure figs directory exists
    os.makedirs('latex/figs', exist_ok=True)
    
    # Generate SSBM
    adj, true_labels = generate_ssbm(n, k, A, B)
    
    # 1. K-Means
    # Note: K-means on adjacency matrix treats each row (neighborhood vector) as a point in R^n
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(adj)
    
    # 2. Spectral Clustering
    # For adjacency matrix, we use affinity='precomputed' or 'nearest_neighbors'
    # Actually, SpectralClustering defaults to 'rbf'. For a graph, we often use 'nearest_neighbors' or 'precomputed'
    spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    spectral_labels = spectral.fit_predict(adj)
    
    # Plotting
    plot_graph_clusters(adj, true_labels, f"Ground Truth SSBM (n={n}, k={k}, A={A}, B={B})", "latex/figs/p1a_ground_truth.png")
    plot_graph_clusters(adj, kmeans_labels, f"K-Means Clustering (ARI={calculate_metrics(true_labels, kmeans_labels):.3f})", "latex/figs/p1a_kmeans.png")
    plot_graph_clusters(adj, spectral_labels, f"Spectral Clustering (ARI={calculate_metrics(true_labels, spectral_labels):.3f})", "latex/figs/p1a_spectral.png")
    
    print(f"P1a: K-Means ARI: {calculate_metrics(true_labels, kmeans_labels):.3f}")
    print(f"P1a: Spectral ARI: {calculate_metrics(true_labels, spectral_labels):.3f}")

if __name__ == "__main__":
    solve_p1a()
