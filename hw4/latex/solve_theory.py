
import numpy as np

def analyze_matrix(name, P):
    print(f"\n--- Analyzing Matrix {name} ---")
    print("P:")
    print(P)
    
    # Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P) # Right eigenvectors for diagonalization P = SDS^-1
    print(f"Eigenvalues of P: {eigenvalues}")
    print("Eigenvectors (columns of S):")
    print(eigenvectors)

    # For stationary dist, we need left eigenvector of P (or right of P.T) for eigenvalue 1
    evals_left, evecs_left = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(evals_left - 1.0))
    pi = evecs_left[:, idx].real
    pi = pi / np.sum(pi) # Normalize
    print(f"\nStationary Distribution (pi = pi * P): {pi}")
    
    # Check convergence of P^n
    print(f"\nChecking convergence of P^n...")
    for n in [10, 20, 50, 100, 1000]:
        Pn = np.linalg.matrix_power(P, n)
        if n == 1000:
            print(f"P^{n}:")
            print(Pn)
            
            # Check if rows are identical to pi
            is_converged = np.allclose(Pn, pi)
            print(f"Converged to stationary distribution? {is_converged}")
            
    # Check if cyclic/periodic
    if not is_converged:
        print("Matrix appears to be periodic/cyclic.")
    else:
        print("Matrix converges.")

def simulate_pandemaniac():
    print(f"\n--- Simulating Pandemaniac (Square Graph) ---")
    # Graph: Square 0-1-2-3-0
    # 0 -- 1
    # |    |
    # 3 -- 2
    adj = {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [2, 0]
    }
    
    # Alternating colors: 0=Red, 1=Blue, 2=Red, 3=Blue
    colors = {0: 'Red', 1: 'Blue', 2: 'Red', 3: 'Blue'}
    
    print(f"Initial State: {colors}")
    
    for t in range(5):
        new_colors = {}
        for node in adj:
            votes = {}
            
            # Self vote (1.5)
            my_color = colors[node]
            votes[my_color] = votes.get(my_color, 0) + 1.5
            
            # Neighbor votes (1.0)
            for neighbor in adj[node]:
                nbr_color = colors[neighbor]
                votes[nbr_color] = votes.get(nbr_color, 0) + 1.0
            
            # Determine majority
            # Total votes = 1.5 (self) + 2 (neighbors) = 3.5
            # Majority > 1.75
            
            majority_color = None
            for c, v in votes.items():
                if v > 1.75:
                    majority_color = c
                    break
            
            if majority_color:
                new_colors[node] = majority_color
            else:
                new_colors[node] = list(votes.keys())[0] # Tie/Keep (shouldn't happen here with strict > 1.75 check on 3.5 total)
                
            print(f"Node {node}: Votes {votes} -> New Color {new_colors[node]}")
            
        colors = new_colors
        print(f"Time {t+1}: {colors}")


if __name__ == "__main__":
    # Problem 4(a)
    Pa = np.array([
        [2/5, 3/10, 3/10],
        [1/5, 3/5, 1/5],
        [7/10, 1/10, 1/5]
    ])
    analyze_matrix("A", Pa)
    
    # Problem 4(b)
    # 0 5/8 0 3/8
    # 1 0 0 0
    # 0 3/8 0 5/8
    # 3/4 0 1/4 0
    Pb = np.array([
        [0, 5/8, 0, 3/8],
        [1, 0, 0, 0],
        [0, 3/8, 0, 5/8],
        [3/4, 0, 1/4, 0]
    ])
    analyze_matrix("B", Pb)
    
    simulate_pandemaniac()
