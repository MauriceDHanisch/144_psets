"""
PageRank algorithm implementation for ranking web pages.

Recall from lecture that the PageRank algorithm is one popular example of an
algorithm that can be applied to rank web pages. When returning search results,
we want to rank pages that are more relevant to the query higher than pages
that are less relevant, and so we turn to algorithms like PageRank to help us
do this.

The PageRank algorithm assigns importance scores to pages based on the link
structure of the web graph. Pages that are linked to by many important pages
receive higher scores.

In this assignment, you will implement the PageRank algorithm yourself below
in the compute_pagerank function. This should be the ONLY implementation
necessary for this part of the assignment. You should NOT call a PageRank
implementation from external libraries (e.g. NetworkX, etc.).

However, PageRank is just one technique we can use to rank search results.
For part of the optional extra credit portion of this question, we invite
you to try out other techniques to rank search results (e.g. your own
ML-based approach, etc.).
"""


def compute_pagerank(graph, damping=0.85, max_iter=10000, tol=1e-8):
    """
    Compute PageRank scores for all nodes in the graph.
    
    Args:
        graph: Dictionary mapping URLs to lists of outbound links.
               Format: {url: [list of linked URLs]}
        damping: Damping factor (default 0.85). Probability of following
                 a link vs. jumping to a random page.
        max_iter: Maximum number of iterations (default 1000)
        tol: Convergence tolerance (default 1e-7)
    
    Returns:
        Dictionary mapping URLs to their PageRank scores.
        Format: {url: pagerank_score}
    """
    
    all_nodes = set(graph.keys())
    for links in graph.values():
        all_nodes.update(links)
    all_nodes = list(all_nodes)
    n = len(all_nodes)
    
    if n == 0:
        return {}

    pr = {node: 1.0 / n for node in all_nodes}
    
    return pr


if __name__ == "__main__":
    # Simple test case
    network1 = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': ['A'],
        'D': ['C']
    }

    # Slightly less trivial test case
    network2 = {
        'A': ['B', 'C', 'D'],
        'B': ['E'],
        'C': ['E'],
        'D': ['E'],
        'E': []
    }

    # Add more networks here if desired

    test_cases = [
        ("Network 1", network1),
        ("Network 2", network2)
    ]

    print("\n" + "=" * 48)
    print("   Running PageRank on Simple Networks   ")
    print("=" * 48 + "\n")

    for name, g in test_cases:
        print(f"{name}")
        print("-" * len(name))
        # Run PageRank on the network with default parameters
        scores = compute_pagerank(g)
        if scores:
            print("{:<6} | {:>10}".format("Node", "PageRank"))
            print("-" * 21)
            for url, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print("{:<6} | {:>10.4f}".format(url, score))
            print()
        else:
            print("PageRank not yet implemented!\n")

