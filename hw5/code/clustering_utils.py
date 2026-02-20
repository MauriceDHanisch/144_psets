import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

def generate_ssbm(n, k, A, B):
    """
    Generate a Symmetric Stochastic Block Model (SSBM) graph.
    n: Total number of nodes
    k: Number of communities
    A: Probability of edge within community
    B: Probability of edge between communities
    """
    labels = np.repeat(np.arange(k), n // k)
    # If n is not divisible by k, distribute the remaining nodes
    remainder = n % k
    if remainder > 0:
        labels = np.concatenate([labels, np.arange(remainder)])
    
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            prob = A if labels[i] == labels[j] else B
            if np.random.rand() < prob:
                adj[i, j] = 1
                adj[j, i] = 1
                
    return adj, labels

def plot_graph_clusters(adj, labels, title, save_path):
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=labels, with_labels=True, cmap=plt.cm.Set1, node_size=200, edge_color='gray', alpha=0.7)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_scatter_clusters(X, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Set1, s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    # Simple accuracy (considering label permutations)
    # For small k, we can find the best permutation, but ARI is generally better for clustering
    return ari
