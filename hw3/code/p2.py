# %% [markdown]|
# # HW 3 Problem 1.2: The Devil is in the Details

# %% [markdown]
# ## Part (a): Preferential Attachment Model

# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

def preferential_attachment(T):
    """
    Generate a network using the preferential attachment model.
    
    Parameters:
    T: int - Total number of nodes (residents)
    
    Returns:
    G: networkx Graph - The generated network
    degrees: list - Degrees of all nodes
    """
    G = nx.Graph()
    
    G.add_edge(0, 1)
    
    for t in range(2, T):
        current_degrees = dict(G.degree())
        total_degree = sum(current_degrees.values())
        
        probabilities = [current_degrees[node] / total_degree for node in G.nodes()]
        
        new_connection = np.random.choice(list(G.nodes()), p=probabilities)
        
        G.add_edge(t, new_connection)
    
    degrees = [G.degree(node) for node in G.nodes()]
    
    return G, degrees

# %%
T = 300
G_pa, degrees_pa = preferential_attachment(T)

sorted_degrees = np.sort(degrees_pa)
unique_degrees, unique_indices = np.unique(sorted_degrees, return_index=True)
ccdf_unique = np.arange(len(sorted_degrees), 0, -1)[unique_indices] / len(sorted_degrees)

plt.figure(figsize=(10, 6))
plt.loglog(unique_degrees, ccdf_unique, 'o', markersize=6, alpha=0.7, label='PA Network')
plt.xlabel('Node Degree')
plt.ylabel('P(Degree > k)')
plt.title('Degree Distribution (CCDF, log-log) - Preferential Attachment (T=300)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('../latex/figs/plot_1.2_a_pa_ccdf.png', dpi=150)
plt.show()

print(f"Preferential Attachment Network (T={T}):")
print(f"Number of nodes: {G_pa.number_of_nodes()}")
print(f"Number of edges: {G_pa.number_of_edges()}")
print(f"Average degree: {np.mean(degrees_pa):.2f}")
print(f"Max degree: {np.max(degrees_pa)}")
print(f"Min degree: {np.min(degrees_pa)}")

# %% [markdown]
# ## Part (b): Configuration Model

# %%
def configuration_model(degree_sequence):
    """
    Generate a network using the configuration model with stub-matching.
    
    Parameters:
    degree_sequence: list - Desired degree sequence
    
    Returns:
    G: networkx Graph - The generated network
    """
    n = len(degree_sequence)
    stubs = []
    
    for node_id, degree in enumerate(degree_sequence):
        stubs.extend([node_id] * degree)
    
    if len(stubs) % 2 != 0:
        return None
    
    np.random.shuffle(stubs)
    
    edges = []
    for i in range(0, len(stubs), 2):
        edges.append((stubs[i], stubs[i+1]))
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    
    return G

# %%
degree_sequence_pa = sorted(degrees_pa, reverse=True)

G_cm = configuration_model(degree_sequence_pa)

degrees_cm = [G_cm.degree(node) for node in G_cm.nodes()]

print(f"\nConfiguration Model Network:")
print(f"Number of nodes: {G_cm.number_of_nodes()}")
print(f"Number of edges: {G_cm.number_of_edges()}")
print(f"Average degree: {np.mean(degrees_cm):.2f}")
print(f"Max degree: {np.max(degrees_cm)}")
print(f"Min degree: {np.min(degrees_cm)}")

# %% [markdown]
# ## Comparing CCDF of Both Models

# %%
sorted_degrees_cm = np.sort(degrees_cm)
unique_degrees_cm, unique_indices_cm = np.unique(sorted_degrees_cm, return_index=True)
ccdf_unique_cm = np.arange(len(sorted_degrees_cm), 0, -1)[unique_indices_cm] / len(sorted_degrees_cm)

plt.figure(figsize=(12, 6))
plt.loglog(unique_degrees, ccdf_unique, 'o', markersize=6, alpha=0.7, label='Preferential Attachment')
plt.loglog(unique_degrees_cm, ccdf_unique_cm, 's', markersize=6, alpha=0.7, label='Configuration Model')
plt.xlabel('Node Degree')
plt.ylabel('P(Degree > k)')
plt.title('Degree Distribution (CCDF, log-log) Comparison')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('../latex/figs/plot_1.2_ab_ccdf_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# ## Part (c): Network Visualization and Comparison

# %%
fig, axes = plt.subplots(2, 3, figsize=(20, 14))

layouts = {
    'Spring': lambda G: nx.spring_layout(G, k=0.5, iterations=50, seed=42),
    'Kamada-Kawai': lambda G: nx.kamada_kawai_layout(G),
    'Circular': lambda G: nx.circular_layout(G),
    'Spring (tight)': lambda G: nx.spring_layout(G, k=0.2, iterations=50, seed=42),
    'Spectral': lambda G: nx.spectral_layout(G),
    'Shell': lambda G: nx.shell_layout(G)
}

layout_names = list(layouts.keys())

for idx, (layout_name, layout_func) in enumerate(layouts.items()):
    pos = layout_func(G_pa)
    ax = axes[idx // 3, idx % 3]
    
    node_sizes = [G_pa.degree(node) * 15 for node in G_pa.nodes()]
    
    nx.draw_networkx_nodes(G_pa, pos, node_size=node_sizes, node_color='lightblue', 
                           alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G_pa, pos, alpha=0.2, width=0.5, ax=ax)
    
    ax.set_title(f'Preferential Attachment - {layout_name}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('../latex/figs/plot_1.2_c_pa_layouts.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(20, 14))

for idx, (layout_name, layout_func) in enumerate(layouts.items()):
    pos = layout_func(G_cm)
    ax = axes[idx // 3, idx % 3]
    
    node_sizes = [G_cm.degree(node) * 15 for node in G_cm.nodes()]
    
    nx.draw_networkx_nodes(G_cm, pos, node_size=node_sizes, node_color='lightcoral', 
                           alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G_cm, pos, alpha=0.2, width=0.5, ax=ax)
    
    ax.set_title(f'Configuration Model - {layout_name}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('../latex/figs/plot_1.2_c_cm_layouts.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Best Layouts - Side by Side Comparison (Spring vs Kamada-Kawai)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

pos_pa_spring = nx.spring_layout(G_pa, k=0.5, iterations=50, seed=42)
node_sizes_pa = [G_pa.degree(node) * 15 for node in G_pa.nodes()]

nx.draw_networkx_nodes(G_pa, pos_pa_spring, node_size=node_sizes_pa, 
                       node_color='lightblue', alpha=0.7, ax=axes[0])
nx.draw_networkx_edges(G_pa, pos_pa_spring, alpha=0.2, width=0.5, ax=axes[0])
axes[0].set_title('Preferential Attachment (Spring Layout)', fontsize=14)
axes[0].axis('off')

pos_cm_spring = nx.spring_layout(G_cm, k=0.5, iterations=50, seed=42)
node_sizes_cm = [G_cm.degree(node) * 15 for node in G_cm.nodes()]

nx.draw_networkx_nodes(G_cm, pos_cm_spring, node_size=node_sizes_cm, 
                       node_color='lightcoral', alpha=0.7, ax=axes[1])
nx.draw_networkx_edges(G_cm, pos_cm_spring, alpha=0.2, width=0.5, ax=axes[1])
axes[1].set_title('Configuration Model (Spring Layout)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../latex/figs/plot_1.2_pa_vs_cm_spring.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

pos_pa_kk = nx.kamada_kawai_layout(G_pa)
node_sizes_pa = [G_pa.degree(node) * 15 for node in G_pa.nodes()]

nx.draw_networkx_nodes(G_pa, pos_pa_kk, node_size=node_sizes_pa, 
                       node_color='lightblue', alpha=0.7, ax=axes[0])
nx.draw_networkx_edges(G_pa, pos_pa_kk, alpha=0.2, width=0.5, ax=axes[0])
axes[0].set_title('Preferential Attachment (Kamada-Kawai Layout)', fontsize=14)
axes[0].axis('off')

pos_cm_kk = nx.kamada_kawai_layout(G_cm)
node_sizes_cm = [G_cm.degree(node) * 15 for node in G_cm.nodes()]

nx.draw_networkx_nodes(G_cm, pos_cm_kk, node_size=node_sizes_cm, 
                       node_color='lightcoral', alpha=0.7, ax=axes[1])
nx.draw_networkx_edges(G_cm, pos_cm_kk, alpha=0.2, width=0.5, ax=axes[1])
axes[1].set_title('Configuration Model (Kamada-Kawai Layout)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../latex/figs/plot_1.2_pa_vs_cm_kk.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
print("\nNetwork Statistics Comparison:")
print("\nPreferential Attachment:")
print(f"  Clustering coefficient: {nx.average_clustering(G_pa):.4f}")
print(f"  Average shortest path length: {nx.average_shortest_path_length(G_pa):.4f}")
print(f"  Density: {nx.density(G_pa):.4f}")
print(f"  Is connected: {nx.is_connected(G_pa)}")

print("\nConfiguration Model:")
print(f"  Clustering coefficient: {nx.average_clustering(G_cm):.4f}")
if nx.is_connected(G_cm):
    print(f"  Average shortest path length: {nx.average_shortest_path_length(G_cm):.4f}")
else:
    largest_cc = max(nx.connected_components(G_cm), key=len)
    G_cm_largest = G_cm.subgraph(largest_cc).copy()
    print(f"  Average shortest path length (largest component): {nx.average_shortest_path_length(G_cm_largest):.4f}")
print(f"  Density: {nx.density(G_cm):.4f}")
print(f"  Is connected: {nx.is_connected(G_cm)}")

# %% [markdown]
# ## Multiple Instances Comparison (PA vs CM, Spring Layout)

# %%
fig, axes = plt.subplots(3, 2, figsize=(18, 20))

for i in range(3):
    G_pa_inst, _ = preferential_attachment(T)
    degree_seq_inst = sorted([G_pa_inst.degree(node) for node in G_pa_inst.nodes()], reverse=True)
    G_cm_inst = configuration_model(degree_seq_inst)
    
    pos_pa = nx.kamada_kawai_layout(G_pa_inst)
    pos_cm = nx.kamada_kawai_layout(G_cm_inst)
    
    node_sizes_pa = [G_pa_inst.degree(node) * 15 for node in G_pa_inst.nodes()]
    node_sizes_cm = [G_cm_inst.degree(node) * 15 for node in G_cm_inst.nodes()]
    
    nx.draw_networkx_nodes(G_pa_inst, pos_pa, node_size=node_sizes_pa, 
                           node_color='lightblue', alpha=0.7, ax=axes[i, 0])
    nx.draw_networkx_edges(G_pa_inst, pos_pa, alpha=0.2, width=0.5, ax=axes[i, 0])
    axes[i, 0].set_title(f'Instance {i+1}: Preferential Attachment', fontsize=12)
    axes[i, 0].axis('off')
    
    nx.draw_networkx_nodes(G_cm_inst, pos_cm, node_size=node_sizes_cm, 
                           node_color='lightcoral', alpha=0.7, ax=axes[i, 1])
    nx.draw_networkx_edges(G_cm_inst, pos_cm, alpha=0.2, width=0.5, ax=axes[i, 1])
    axes[i, 1].set_title(f'Instance {i+1}: Configuration Model', fontsize=12)
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('../latex/figs/plot_1.2_c_pa_vs_cm_multiple.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
