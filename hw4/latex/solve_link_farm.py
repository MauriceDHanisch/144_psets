
import numpy as np

def compute_pagerank(adj, alpha=0.85, tol=1e-9):
    """
    adj: dict {source: [destinations]}
    If node has no out-links, it has a self-loop (as per HW spec).
    """
    nodes = sorted(list(adj.keys()))
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    P = np.zeros((n, n))
    
    for u in nodes:
        dests = adj[u]
        if not dests: # Sink -> Self-loop
            dests = [u]
            
        prob = 1.0 / len(dests)
        for v in dests:
            if v in node_to_idx:
                P[node_to_idx[u], node_to_idx[v]] = prob
            
    # PageRank Iteration
    # pi = alpha * pi * P + (1-alpha)/n * 1
    
    pi = np.ones(n) / n
    err = 1.0
    while err > tol:
        new_pi = alpha * np.dot(pi, P) + (1 - alpha) / n
        err = np.linalg.norm(new_pi - pi, 1)
        pi = new_pi
        
    return {nodes[i]: pi[i] for i in range(n)}

def run_scenarios():
    N_WEB = 100
    alpha = 0.85
    
    # Base Web: Cycle 0->1->...->N-1->0
    # This gives uniform PR for the base web, making analysis clear.
    base_adj = {i: [(i+1)%N_WEB] for i in range(N_WEB)}
    
    print(f"--- Scenario A: Add X (Isolated) ---")
    adj_a = base_adj.copy()
    X = N_WEB
    adj_a[X] = [] # Sink -> Self loop
    pr_a = compute_pagerank(adj_a, alpha)
    print(f"PR(X): {pr_a[X]:.6f}")
    
    print(f"\n--- Scenario B: Add X <- Y (Y isolated) ---")
    adj_b = base_adj.copy()
    X = N_WEB
    Y = N_WEB + 1
    adj_b[X] = [] # Sink (Self)
    adj_b[Y] = [X] # Y points to X
    pr_b = compute_pagerank(adj_b, alpha)
    print(f"PR(X): {pr_b[X]:.6f}")
    print(f"PR(Y): {pr_b[Y]:.6f}")
    
    print(f"\n--- Scenario C: Add X, Y, Z (Maximize X) ---")
    X, Y, Z = N_WEB, N_WEB + 1, N_WEB + 2
    
    # Topology 1: Y->X, Z->X, X->Self
    adj_c1 = base_adj.copy()
    adj_c1[X] = []
    adj_c1[Y] = [X]
    adj_c1[Z] = [X]
    pr_c1 = compute_pagerank(adj_c1, alpha)
    print(f"1. Y->X, Z->X, X Absorb: PR(X)={pr_c1[X]:.6f}")

    # Topology 2: Cycle X->Y->Z->X
    adj_c2 = base_adj.copy()
    adj_c2[X] = [Y]
    adj_c2[Y] = [Z]
    adj_c2[Z] = [X]
    pr_c2 = compute_pagerank(adj_c2, alpha)
    print(f"2. Cycle X->Y->Z->X:    PR(X)={pr_c2[X]:.6f}")
    
    # Topology 3: Y->X, Z->X, X->Y, X->Z (Mutual reinforcement?)
    # If X points to Y and Z, it loses mass.
    # We want X to KEEP mass. X should be a sink.
    
    # Topology 3: Y->X, Z->X, Y->Z ??
    adj_c3 = base_adj.copy()
    adj_c3[X] = []
    adj_c3[Y] = [X] # Y dumps to X
    adj_c3[Z] = [X, Y] # Z divides
    pr_c3 = compute_pagerank(adj_c3, alpha)
    print(f"3. Y->X, Z->{X,Y}, X Abs: PR(X)={pr_c3[X]:.6f}")
    
    # Topology 4: Y->X, Z->Y (Funnel)
    # Z -> Y -> X (loop)
    adj_c4 = base_adj.copy()
    adj_c4[X] = []
    adj_c4[Y] = [X]
    adj_c4[Z] = [Y]
    pr_c4 = compute_pagerank(adj_c4, alpha)
    print(f"4. Funnel Z->Y->X Abs:   PR(X)={pr_c4[X]:.6f}")

if __name__ == "__main__":
    run_scenarios()
