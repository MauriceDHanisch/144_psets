import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def linear_payoffs(G, c, graph_type='undirected'):
    n = G.number_of_nodes()
    payoffs = np.zeros(n)
    for i in range(n):
        if graph_type == 'undirected':
            ni = G.degree(i)
            reach = len(nx.node_connected_component(G, i)) - 1
        else:
            ni = G.out_degree(i)
            reach = len(nx.descendants(G, i))
            
        payoffs[i] = -c * ni + reach
    return payoffs

def connections_payoffs(G, c, delta, graph_type='undirected'):
    n = G.number_of_nodes()
    payoffs = np.zeros(n)
    path_lengths = dict(nx.shortest_path_length(G))
    for i in range(n):
        if graph_type == 'undirected':
            ni = G.degree(i)
        else:
            ni = G.out_degree(i)
            
        benefit = 0
        for j in range(n):
            if i != j:
                if j in path_lengths[i]:
                    d = path_lengths[i][j]
                    benefit += delta ** d
        
        payoffs[i] = -c * ni + benefit
    return payoffs

def get_possible_actions(G, u, graph_type='undirected'):
    actions = [('none', None)]
    n = G.number_of_nodes()
    
    if graph_type == 'undirected':
        neighbors = set(G.neighbors(u))
    else:
        neighbors = set(G.successors(u))
        
    non_neighbors = set(range(n)) - neighbors - {u}
    
    for w in non_neighbors:
        actions.append(('add', w))
        
    for v in neighbors:
        actions.append(('remove', v))
        
    for v in neighbors:
        for w in non_neighbors:
            actions.append(('swap', (v, w)))
            
    return actions

def apply_action(G, u, action, graph_type='undirected'):
    G_new = G.copy()
    act_type, params = action
    
    if act_type == 'none':
        pass
    elif act_type == 'add':
        G_new.add_edge(u, params)
    elif act_type == 'remove':
        G_new.remove_edge(u, params)
    elif act_type == 'swap':
        v, w = params
        G_new.remove_edge(u, v)
        G_new.add_edge(u, w)
        
    return G_new

def run_simulation(init_G, c, model='linear', delta=0.5, graph_type='undirected', max_iters=500):
    G = init_G.copy()
    n = G.number_of_nodes()
    
    history = [tuple(sorted(G.edges()))]
    
    if model == 'linear':
        eff = sum(linear_payoffs(G, c, graph_type))
    else:
        eff = sum(connections_payoffs(G, c, delta, graph_type))
    efficiencies = [eff]
    
    for t in range(max_iters):
        u = t % n
        
        actions = get_possible_actions(G, u, graph_type)
        
        best_payoff = -float('inf')
        best_actions = []
        
        for action in actions:
            G_new = apply_action(G, u, action, graph_type)
            if model == 'linear':
                payoff = linear_payoffs(G_new, c, graph_type)[u]
            else:
                payoff = connections_payoffs(G_new, c, delta, graph_type)[u]
                
            if payoff > best_payoff + 1e-9:
                best_payoff = payoff
                best_actions = [action]
            elif abs(payoff - best_payoff) < 1e-9:
                best_actions.append(action)
                
        none_in_best = False
        for a in best_actions:
            if a[0] == 'none':
                none_in_best = True
                break
                
        if none_in_best:
            chosen_action = ('none', None)
        else:
            chosen_action = best_actions[np.random.choice(len(best_actions))]
            
        G = apply_action(G, u, chosen_action, graph_type)
        history.append(tuple(sorted(G.edges())))
        
        if model == 'linear':
            eff = sum(linear_payoffs(G, c, graph_type))
        else:
            eff = sum(connections_payoffs(G, c, delta, graph_type))
        efficiencies.append(eff)
        
        if t >= n:
            edges_same = True
            for i in range(1, n+1):
                if history[-1] != history[-(i+1)]:
                    edges_same = False
                    break
            if edges_same:
                break
                
    return G, history, efficiencies

def calculate_efficiency(G, c, model, delta, graph_type):
    if model == 'linear':
        payoffs = linear_payoffs(G, c, graph_type)
    else:
        payoffs = connections_payoffs(G, c, delta, graph_type)
    return sum(payoffs)

def generate_inits(n, graph_type):
    inits = {}
    if graph_type == 'undirected':
        inits['Empty'] = nx.empty_graph(n)
        inits['Complete'] = nx.complete_graph(n)
        inits['Star'] = nx.star_graph(n-1)
        inits['Cycle'] = nx.cycle_graph(n)
    else:
        inits['Empty'] = nx.empty_graph(n, create_using=nx.DiGraph)
        inits['Complete'] = nx.complete_graph(n, create_using=nx.DiGraph)
        star_out = nx.DiGraph()
        star_out.add_nodes_from(range(n))
        for i in range(1, n):
            star_out.add_edge(0, i)
        inits['Star_Out'] = star_out
        
        star_in = nx.DiGraph()
        star_in.add_nodes_from(range(n))
        for i in range(1, n):
            star_in.add_edge(i, 0)
        inits['Star_In'] = star_in
        
        cycle = nx.DiGraph()
        cycle.add_nodes_from(range(n))
        for i in range(n):
            cycle.add_edge(i, (i+1)%n)
        inits['Cycle'] = cycle
    return inits

def plot_graph(G, title, filename):
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, 
            arrows=isinstance(G, nx.DiGraph), edge_color='gray')
    plt.title(title)
    plt.savefig(f'/Users/mhanisch/projects/psets/q2/144/hw6/latex/figs/{filename}')
    plt.close()

def plot_convergence(efficiencies, title, filename):
    plt.figure(figsize=(6,4))
    plt.plot(efficiencies, marker='o', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Total Efficiency')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'/Users/mhanisch/projects/psets/q2/144/hw6/latex/figs/{filename}', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    n = 9
    out_lines = []
    
    # 1. Undirected Linear
    out_lines.append("=== Part 1: Undirected Linear ===")
    c_vals = [0.5, 4.0, 10.0]
    c_labels = ['c_lt_1', '1_lt_c_lt_n-1', 'c_gt_n-1']
    for c, label in zip(c_vals, c_labels):
        out_lines.append(f"\\nRegime: {label} (c = {c})")
        inits = generate_inits(n, 'undirected')
        for init_name, init_G in inits.items():
            G_eq, history, efficiencies = run_simulation(init_G, c, model='linear', graph_type='undirected')
            eff = calculate_efficiency(G_eq, c, 'linear', delta=0, graph_type='undirected')
            out_lines.append(f"  Init: {init_name:<10} | Steps: {len(history)-1:<4} | Efficiency: {eff:<6.2f} | Num Edges: {G_eq.number_of_edges()}")
            plot_graph(G_eq, f"Undirected Linear, {label}, Init: {init_name}", f"undirected_linear_{label}_{init_name}.png")
            plot_convergence(efficiencies, f"Efficiency: Undirected Linear, {label}, Init: {init_name}", f"conv_undirected_linear_{label}_{init_name}.png")

    # 2. Directed Linear
    out_lines.append("\\n=== Part 2: Directed Linear ===")
    for c, label in zip(c_vals, c_labels):
        out_lines.append(f"\\nRegime: {label} (c = {c})")
        inits = generate_inits(n, 'directed')
        for init_name, init_G in inits.items():
            G_eq, history, efficiencies = run_simulation(init_G, c, model='linear', graph_type='directed')
            eff = calculate_efficiency(G_eq, c, 'linear', delta=0, graph_type='directed')
            out_lines.append(f"  Init: {init_name:<10} | Steps: {len(history)-1:<4} | Efficiency: {eff:<6.2f} | Num Edges: {G_eq.number_of_edges()}")
            plot_graph(G_eq, f"Directed Linear, {label}, Init: {init_name}", f"directed_linear_{label}_{init_name}.png")
            plot_convergence(efficiencies, f"Efficiency: Directed Linear, {label}, Init: {init_name}", f"conv_directed_linear_{label}_{init_name}.png")

    # 3. Undirected Connections
    out_lines.append("\\n=== Part 3: Undirected Connections ===")
    delta = 0.5
    # thresholds: delta - delta**2 = 0.25
    # delta + (n/2-1)*delta**2 = 0.5 + (3.5)*0.25 = 1.375
    c_vals_conn = [0.1, 0.8, 2.0]
    c_labels_conn = ['c_small', 'c_medium', 'c_large']
    for c, label in zip(c_vals_conn, c_labels_conn):
        out_lines.append(f"\\nRegime: {label} (c = {c})")
        inits = generate_inits(n, 'undirected')
        for init_name, init_G in inits.items():
            G_eq, history, efficiencies = run_simulation(init_G, c, model='connections', delta=delta, graph_type='undirected')
            eff = calculate_efficiency(G_eq, c, 'connections', delta=delta, graph_type='undirected')
            out_lines.append(f"  Init: {init_name:<10} | Steps: {len(history)-1:<4} | Efficiency: {eff:<6.2f} | Num Edges: {G_eq.number_of_edges()}")
            plot_graph(G_eq, f"Undirected Connections, {label}, Init: {init_name}", f"undirected_conn_{label}_{init_name}.png")
            plot_convergence(efficiencies, f"Efficiency: Undirected Connections, {label}, Init: {init_name}", f"conv_undirected_conn_{label}_{init_name}.png")

    with open('/Users/mhanisch/projects/psets/q2/144/hw6/code/simulate_networks.py.out', 'w') as f:
        f.write("\\n".join(out_lines))
