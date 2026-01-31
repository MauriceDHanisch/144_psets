import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

SEED = 42

def ER_G(n, p, seed=42): 
    rng = np.random.RandomState(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < p:
                G.add_edge(i, j)
    return G
        
G_er_40 = ER_G(40, 0.23, SEED)

# Plot 1: Spring Layout (Effective)
plt.figure(figsize=(8, 8), dpi=300)
pos = nx.spring_layout(G_er_40, k=0.5, seed=SEED)
nx.draw(G_er_40, pos, node_size=60, node_color='skyblue', edge_color='gray', width=0.5, alpha=0.7)
plt.title("Erdős-Rényi (n=40, p=0.23) - Spring Layout")
os.makedirs('latex/figs', exist_ok=True)
plt.savefig('latex/figs/2a_ER_spring.png', bbox_inches='tight')
plt.close()

# Plot 2: Circular Layout (Contrasting)
plt.figure(figsize=(8, 8), dpi=200)
pos_circ = nx.circular_layout(G_er_40)
nx.draw(G_er_40, pos_circ, node_size=50, node_color='skyblue', edge_color='gray', width=0.3, alpha=0.5)
plt.title("Erdős-Rényi (n=40, p=0.23) - Circular Layout")
plt.savefig('latex/figs/2a_ER_circular_contrast.png', bbox_inches='tight')
plt.close()

# ER with n=100 to show dense structure better
G_er_100 = ER_G(100, 0.23, SEED)

plt.figure(figsize=(10, 10), dpi=300)
pos = nx.spring_layout(G_er_100, k=0.8, seed=SEED)
# Balanced alpha and width for high density
nx.draw(G_er_100, pos, node_size=40, node_color='steelblue', edge_color='gray', width=0.2, alpha=0.3)
plt.title("Erdős-Rényi (n=100, p=0.23) - Structure View")
plt.savefig('latex/figs/2a_ER_100.png', bbox_inches='tight')
plt.close()

import matplotlib.lines as mlines

def SSBM_G(n, k, A, B, seed=42):
    rng = np.random.RandomState(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Assign clusters
    nodes = np.array(range(n))
    rng.shuffle(nodes)
    cluster_size = n // k
    labels = np.zeros(n, dtype=int)
    for i in range(k):
        labels[nodes[i*cluster_size:(i+1)*cluster_size]] = i
        
    for i in range(n):
        for j in range(i + 1, n):
            prob = A if labels[i] == labels[j] else B
            if rng.rand() < prob:
                G.add_edge(i, j)
                
    return G, labels

def plot_colored_ssbm(G, labels, title, filename, layout='spring'):
    plt.figure(figsize=(10, 10), dpi=300)
    
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=SEED, k=0.5)
    else:
        pos = nx.random_layout(G, seed=SEED)
    
    # Color nodes by cluster
    colors = ['#66c2a5', '#fc8d62', '#8da0cb'] # Set 2 colors
    node_colors = [colors[labels[i]] for i in G.nodes()]
    
    # Color edges: light gray for intra-cluster, salmon for inter-cluster
    edge_colors = []
    for u, v in G.edges():
        if labels[u] == labels[v]:
            edge_colors.append('#cccccc') # Light gray intra
        else:
            edge_colors.append('#e78ac3') # Pinkish inter
            
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color=edge_colors, alpha=0.8)
    
    # Create legend
    cluster_handles = [mlines.Line2D([], [], color='white', marker='o', markerfacecolor=c, markersize=10, label=f'Cluster {i}') for i, c in enumerate(colors)]
    edge_handles = [
        mlines.Line2D([], [], color='#cccccc', linewidth=2, label='Intra-cluster'),
        mlines.Line2D([], [], color='#e78ac3', linewidth=2, label='Inter-cluster')
    ]
    plt.legend(handles=cluster_handles + edge_handles, loc='upper right')
    
    plt.title(title)
    plt.axis('off')
    os.makedirs('latex/figs', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Generate and plot SSBM
G_ssbm_30, labels_30 = SSBM_G(30, 3, 0.8, 0.05, SEED)
plot_colored_ssbm(G_ssbm_30, labels_30, "SSBM (n=30, k=3) - Clustered View", "latex/figs/2b_SSBM_spring.png", layout='spring')

# Random layout for contrast - WITH edge coloring
plot_colored_ssbm(G_ssbm_30, labels_30, "SSBM (n=30) - Random Layout (Contrast)", "latex/figs/2b_SSBM_random_contrast.png", layout='random')

# Larger SSBM check
G_ssbm_100, labels_100 = SSBM_G(100, 3, 0.5, 0.05, SEED)
plot_colored_ssbm(G_ssbm_100, labels_100, "SSBM (n=100) - Clustered", "latex/figs/2b_SSBM_100.png", layout='spring')

# SSBM with n=100 to show clustering better
G_ssbm_100, labels_100 = SSBM_G(100, 3, 0.7, 0.1, SEED)
plot_colored_ssbm(G_ssbm_100, labels_100, "SSBM (n=100, k=3) - Structure View", "2b_SSBM_100")

import pickle

with open('data/caltech_web_graph.pkl', 'rb') as f:
    G_web = pickle.load(f)

# Use list to index nodes
nodes = list(G_web.nodes())
G_web_100 = G_web.subgraph(nodes[:100])
G_web_300 = G_web.subgraph(nodes[:300])

# 1. Spring Layout for n=100
plt.figure(figsize=(8, 8), dpi=300)
pos_spring = nx.spring_layout(G_web_100, k=0.5, seed=SEED)
nx.draw(G_web_100, pos_spring, node_size=60, node_color='skyblue', edge_color='gray', width=0.5, alpha=0.7)
plt.title("Web Graph (n=100) - Spring Layout")
os.makedirs('latex/figs', exist_ok=True)
plt.savefig('latex/figs/2c_Web100_spring.png', bbox_inches='tight')
plt.close()

# 2. Radial/Shell Layout for n=100 (Contrast)
plt.figure(figsize=(8, 8), dpi=300)
pos_shell = nx.shell_layout(G_web_100)
nx.draw(G_web_100, pos_shell, node_size=60, node_color='skyblue', edge_color='gray', width=0.5, alpha=0.7)
plt.title("Web Graph (n=100) - Shell Layout (Contrast)")
plt.savefig('latex/figs/2c_Web100_radial_contrast.png', bbox_inches='tight')
plt.close()

# Web 300 - Spring with balanced alpha/width
plt.figure(figsize=(8, 8), dpi=300)
pos = nx.spring_layout(G_web_300, k=0.4, seed=SEED)
# Very low alpha and thin lines for dense web graph
nx.draw(G_web_300, pos, node_size=60, node_color='skyblue', edge_color='gray', width=0.1, alpha=0.3)
plt.title("Web Graph (n=300) - Spring Layout (Alpha/Width adapted)")
plt.savefig('latex/figs/2c_Web300_spring.png', bbox_inches='tight')
plt.close()