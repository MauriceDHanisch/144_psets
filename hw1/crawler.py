from fetcher3 import fetch_links
import networkx as nx
import matplotlib.pyplot as plt
import concurrent.futures
import pickle

start_url = "https://www.caltech.edu"
num_nodes_limit = 2000
MAX_WORKERS = 5  # Polite parallelism (approx 5 concurrent requests)

G = nx.DiGraph()
G.add_node(start_url)

to_visit = [start_url]
visited = set()

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    while len(visited) < num_nodes_limit and to_visit:
        # 1. Select a batch of URLs to visit (BFS)
        batch = []
        while len(batch) < MAX_WORKERS and to_visit:
            url = to_visit.pop(0)
            if url not in visited:
                batch.append(url)
                visited.add(url)
        
        if not batch:
            break

        # 2. Parallel Fetch
        future_to_url = {executor.submit(fetch_links, url): url for url in batch}
        
        # 3. Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            current_url = future_to_url[future]
            try:
                links = future.result()
                if links:
                    for link in links:
                        if "caltech.edu" in link:
                            # Add edge (safe: main thread does this)
                            G.add_edge(current_url, link)
                            # Add to queue if not seen (approximate check, set handles exact)
                            if link not in visited:
                                to_visit.append(link)
            except Exception as e:
                pass # safely ignore fetch errors
            print(f"Crawled: {len(visited)} | Nodes: {G.number_of_nodes()} | Queue: {len(to_visit)}", end='\r')

unvisited_nodes = [node for node in G.nodes() if node not in visited]
G.remove_nodes_from(unvisited_nodes)

print(f"\nFinal Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save the graph
with open('caltech_web_graph.pkl', 'wb') as f:
    pickle.dump(G, f)
print("Graph saved to caltech_web_graph.pkl")

# plot the graph
plt.figure(figsize=(10, 10))
nx.draw(G, 
        pos=nx.spring_layout(G, k=0.15, iterations=20), # k controls spacing
        node_size=10, 
        width=0.1, 
        arrowsize=5,
        with_labels=False, 
        node_color='blue', 
        edge_color='gray', 
        alpha=0.6)
plt.title("Caltech Web Crawl")
plt.axis('off')
plt.savefig('caltech_web_graph.png', dpi=500)