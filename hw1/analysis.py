import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import os

# Graph loading moved to analyze_dataset function

def plot_histogram(data, title, xlabel, ylabel, filename):
    plt.figure()
    plt.hist(data, bins=range(min(data), max(data) + 2, 5), align='left', rwidth=0.8)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def plot_ccdf(data, title, xlabel, ylabel, filename):
    # Filter out zeros for log-log plot (reciprocal of infinite is 0, log(0) is undef)
    # or just shift them if 0 is meaningful. For degrees, 0 is possible.
    # Usually we plot data > 0 for power law checks.
    data = np.array(data)
    data = data[data > 0]
    
    if len(data) == 0:
        print(f"Warning: No positive data for {title}")
        return

    data_sorted = np.sort(data)
    # Calculate CCDF: P(X >= x)
    # rank i (0-based) corresponds to N-i elements >= x
    # y = (N - i) / N
    n = len(data_sorted)
    y = np.arange(n, 0, -1) / n
    
    plt.figure()
    plt.loglog(data_sorted, y, marker='.', linestyle='none')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(filename)
    plt.close()

def analyze_dataset(pickle_file, label, file_prefix=''):
    print(f"\n--- Analyzing {label} ({pickle_file}) ---")
    
    try:
        with open(pickle_file, 'rb') as f:
            G = pickle.load(f)
        print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    except FileNotFoundError:
        print(f"File {pickle_file} not found. Skipping.")
        return

    # 1.3 Histograms & 1.4 CCDF
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    in_degrees = [G.in_degree(n) for n in G.nodes()]

    prefix = file_prefix
    
    plot_histogram(out_degrees, f"Out-Degree Distribution ({label})", "Out-Degree", "Frequency", f"{prefix}hist_out_degree.png")
    plot_histogram(in_degrees, f"In-Degree Distribution ({label})", "In-Degree", "Frequency", f"{prefix}hist_in_degree.png")

    plot_ccdf(out_degrees, f"CCDF of Out-Degree ({label})", "Out-Degree", "P(X >= x)", f"{prefix}ccdf_out_degree.png")
    plot_ccdf(in_degrees, f"CCDF of In-Degree ({label})", "In-Degree", "P(X >= x)", f"{prefix}ccdf_in_degree.png")

    # 1.5 Clustering Coefficients - Treat as undirected
    G_undirected = G.to_undirected()
    
    # Handle disconnected components if necessary for diameter
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_comp = G_undirected.subgraph(largest_cc)
        print(f"Graph is not connected. Using largest component ({len(largest_cc)} nodes) for diameter.")
    else:
        G_comp = G_undirected

    avg_clustering = nx.average_clustering(G_undirected)
    overall_clustering = nx.transitivity(G_undirected)

    print(f"Average Clustering Coefficient: {avg_clustering}")
    print(f"Overall Clustering Coefficient: {overall_clustering}")

    # 1.6 Diameter
    try:
        diameter = nx.diameter(G_comp)
        avg_diameter = nx.average_shortest_path_length(G_comp)
        print(f"Maximal Diameter: {diameter}")
        print(f"Average Diameter: {avg_diameter}")
    except Exception as e:
        print(f"Could not calculate diameter: {e}")
        diameter = "N/A"
        avg_diameter = "N/A"

    # Write results
    mode = 'a'
    with open("analysis_results.txt", mode) as f:
        f.write(f"\n--- {label} ---\n")
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n")
        f.write(f"Average Clustering Coefficient: {avg_clustering:.4f}\n")
        f.write(f"Overall Clustering Coefficient: {overall_clustering:.4f}\n")
        f.write(f"Maximal Diameter: {diameter}\n")
        f.write(f"Average Diameter: {avg_diameter if isinstance(avg_diameter, str) else f'{avg_diameter:.4f}'}\n")

# Clear results file
with open("analysis_results.txt", "w") as f:
    f.write("Analysis Report Summary\n")

# Run analysis for both
# Parallel (original names maintained by empty prefix)
analyze_dataset('caltech_web_graph.pkl', 'Parallel', '')
# Sequential (prefixed names)
analyze_dataset('caltech_web_graph_np.pkl', 'Sequential', 'seq_')