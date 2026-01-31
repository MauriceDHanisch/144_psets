from fetcher3 import fetch_links
import networkx as nx
import matplotlib.pyplot as plt
import pickle

start_url = "https://www.caltech.edu"
num_nodes_limit = 2000

G = nx.DiGraph()
G.add_node(start_url)

to_visit = [start_url]
visited = set()

while len(visited) < num_nodes_limit and to_visit:
    current_url = to_visit.pop(0)
    
    if current_url in visited:
        continue
    
    visited.add(current_url)
    
    try:
        links = fetch_links(current_url)
        if links:
            for link in links:
                if "caltech.edu" in link:
                    G.add_edge(current_url, link)
                    if link not in visited:
                        to_visit.append(link)
    except Exception:
        pass # Ignore fetch errors
        
    print(f"Crawled: {len(visited)} | Nodes: {G.number_of_nodes()} | Queue: {len(to_visit)}", end='\r')

# Clean up nodes that weren't visited
unvisited_nodes = [node for node in G.nodes() if node not in visited]
G.remove_nodes_from(unvisited_nodes)

print(f"\nFinal Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save the graph with a different name
with open('caltech_web_graph_np.pkl', 'wb') as f:
    pickle.dump(G, f)
print("Graph saved to caltech_web_graph_np.pkl")

# Plot the graph
plt.figure(figsize=(10, 10))
nx.draw(G, 
        pos=nx.spring_layout(G, k=0.15, iterations=20),
        node_size=10, 
        width=0.1, 
        arrowsize=5,
        with_labels=False, 
        node_color='green', 
        edge_color='gray', 
        alpha=0.6)
plt.title("Caltech Web Crawl (Sequential)")
plt.axis('off')
plt.savefig('caltech_web_graph_np.png', dpi=500)
