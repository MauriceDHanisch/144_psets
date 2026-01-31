import numpy as np
import networkx as nx
from tqdm import tqdm

n = 1000
k = 10
p = 0.1
n_samples = 10000

def d_ring(u, v, n=1000):
    dist = np.abs(u - v)
    return np.minimum(dist, n - dist)

def greedy_length(G, u, v):
    length = 0
    
    while u != v:
        if length >= n: # Prevent infinite loops
            print(f"Warning: exceeded max length in greedy search from {u} to {v}")
            if (np.abs(u - v) != 1):
                print(f"Stuck at node {u} trying to reach {v}")
            return np.inf
        neighbors = np.array(list(nx.neighbors(G, u)))
        d_rings = d_ring(neighbors, v)
        u = neighbors[np.argmin(d_rings)]
        length += 1

    return length

G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=42)
print("is connected:", nx.is_connected(G))

uvs = np.random.choice(list(range(n)), (n_samples, 2), replace=True) # This doesn't exclude (u, u)

shortest_lens = []
greedy_lens = []
num_failed = 0
for i in tqdm(range(len(uvs))):
    if np.abs(uvs[i, 0] - uvs[i, 1]) == 1:
        print(f"neighbor nodes {uvs[i, 0]}, {uvs[i, 1]}")
    shortest_lens.append(nx.shortest_path_length(G, uvs[i, 0], uvs[i, 1]))
    greedy_lens.append(greedy_length(G, uvs[i, 0], uvs[i, 1]))
    if greedy_lens[-1] == np.inf:
        num_failed += 1
print(f"Avg. shortest length: {np.mean(shortest_lens)}")
print(f"Avg. greedy length: {np.mean(greedy_lens)}")
print(f"Number of failed greedy searches: {num_failed} out of {n_samples} -> probability {num_failed / n_samples}")



