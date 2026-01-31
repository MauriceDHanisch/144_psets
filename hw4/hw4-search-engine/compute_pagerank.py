#!/usr/bin/env python3
"""
Compute PageRank scores for the crawled web graph and save them.

Run this after crawling to compute PageRank scores for all pages.
The scores will be saved to data/pagerank.json for use by the search engine.
"""

import json
import sys
from src import compute_pagerank


def main():
    data_dir = 'data'
    
    print("Loading web graph...")
    try:
        with open(f'{data_dir}/graph.json', 'r') as f:
            graph = json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_dir}/graph.json not found!")
        print("Please run the crawler first to generate the web graph.")
        print("  python crawl.py")
        sys.exit(1)
    
    print(f"Graph loaded: {len(graph)} pages\n")
    
    print("Computing PageRank scores...")
    print("(This may take a moment...)\n")
    
    scores = compute_pagerank(graph)
    
    if scores is None or len(scores) == 0:
        print("Error: PageRank computation returned no results.")
        print("Make sure you've implemented compute_pagerank in src/pagerank.py!")
        sys.exit(1)
    
    # Save scores
    output_path = f'{data_dir}/pagerank.json'
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"PageRank scores computed and saved to {output_path}")
    print(f"Total pages: {len(scores)}")
    
    # Show top 10 pages by PageRank for debugging
    top_pages = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 pages by PageRank:")
    for i, (url, score) in enumerate(top_pages, 1):
        print(f"  {i:2d}. {score:.6f}  {url}")


if __name__ == "__main__":
    main()
