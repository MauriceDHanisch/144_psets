import os
import matplotlib.pyplot as plt
from braess_simulation import BraessSimulator

def solve_p2():
    sim = BraessSimulator(num_drivers=4000)
    os.makedirs('latex/figs', exist_ok=True)
    
    # Part (a)
    final_dist_a, history_a = sim.run_simulation(part='a', iterations=1000)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_a['top'], label='S -> A -> E')
    plt.plot(history_a['bottom'], label='S -> B -> E')
    plt.plot(history_a['average'], 'k--', label='Average Travel Time', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Travel Time')
    plt.title('Braess Paradox Part (a): Initial Equilibrium')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('latex/figs/p2a_plot.png')
    plt.close()
    
    print(f"P2a Final Distribution: Top={final_dist_a[0]}, Bottom={final_dist_a[1]}")
    c1, c2 = sim.get_costs_part_a(*final_dist_a)
    print(f"P2a Final Costs: Path1={c1:.2f}, Path2={c2:.2f}, Avg={(final_dist_a[0]*c1 + final_dist_a[1]*c2)/4000:.2f}")

    # Part (b)
    final_dist_b, history_b = sim.run_simulation(part='b', iterations=1000, initial_dist=final_dist_a)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_b['top'], label='S -> A -> E')
    plt.plot(history_b['bottom'], label='S -> B -> E')
    plt.plot(history_b['middle'], label='S -> A -> B -> E (New Path)')
    plt.plot(history_b['average'], 'k--', label='Average Travel Time', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Travel Time')
    plt.title('Braess Paradox Part (b): Adding New Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('latex/figs/p2b_plot.png')
    plt.close()
    
    print(f"P2b Final Distribution: Top={final_dist_b[0]}, Bottom={final_dist_b[1]}, Middle={final_dist_b[2]}")
    c1, c2, c3 = sim.get_costs_part_b(*final_dist_b)
    print(f"P2b Final Costs: Path1={c1:.2f}, Path2={c2:.2f}, Path3={c3:.2f}, Avg={(final_dist_b[0]*c1 + final_dist_b[1]*c2 + final_dist_b[2]*c3)/4000:.2f}")

if __name__ == "__main__":
    solve_p2()
