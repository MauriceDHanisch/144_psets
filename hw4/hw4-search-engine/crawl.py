#!/usr/bin/env python3
"""
Run the web crawler to build the web graph.

This script crawls the Caltech domain and saves the resulting graph
to data/graph.json for use by the search engine.
"""

import sys
from src import Crawler


def main():
    seed_url = "http://www.caltech.edu/"
    max_pages = 5000
    delay = 0.001
    
    # Accept command-line arguments for seed URL and max pages
    if len(sys.argv) > 1:
        seed_url = sys.argv[1]
    if len(sys.argv) > 2:
        max_pages = int(sys.argv[2])
    
    print("=" * 80)
    print("Caltech Web Crawler")
    print("=" * 80)
    print(f"Seed URL: {seed_url}")
    print(f"Max pages: {max_pages}")
    print(f"Delay: {delay}s\n")
    
    crawler = Crawler(seed_url=seed_url, max_pages=max_pages)
    crawler.crawl(delay=delay)
    crawler.save_results()
    
    print("\nCrawl and indexing complete! Next step:")
    print("  python compute_pagerank.py  (compute PageRank scores)")


if __name__ == "__main__":
    main()
