#!/usr/bin/env python3
"""
Command-line interface for the Bernoulli search engine.

Usage:
    python search.py                    # Interactive mode
    python search.py "your query"       # Single query mode

Remember that for extra credit, you may wish to build a
user interface for the search engine.
"""
import sys
import re
import os
from src import SearchEngine

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ORANGE = '\033[38;2;255;128;0m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


def highlight_text(text, query_words):
    """Highlight query word(s) in text."""
    if not query_words:
        return text
    # Case-insensitive patterning
    pattern = '|'.join(re.escape(word) for word in query_words)
    if not pattern:
        return text
    
    # Highlight matches
    def highlight_match(match):
        return f"{Colors.YELLOW}{Colors.BOLD}{match.group(0)}{Colors.END}"
    
    return re.sub(f'({pattern})', highlight_match, text, flags=re.IGNORECASE)


def format_url(url, max_length=70):
    """Format URL, truncating if too long."""
    if len(url) <= max_length:
        return url
    return url[:max_length-3] + '...'


def print_results(results, query):
    """Print search results in a nice format."""
    query_words = re.findall(r'\b[a-z0-9]+\b', query.lower())
    
    if not results:
        print(f"\n{Colors.RED}No results found for: {Colors.BOLD}'{query}'{Colors.END}\n")
        print(f"{Colors.DIM}Try different keywords or check spelling.{Colors.END}\n")
        return
    
    # Header
    result_count = len(results)
    print(f"\n{Colors.GREEN}{Colors.BOLD}Found {result_count} result{'s' if result_count != 1 else ''}{Colors.END} "
          f"{Colors.DIM}for: '{query}'{Colors.END}\n")
    print(f"{Colors.DIM}{'─' * 80}{Colors.END}")
    
    # Results
    for i, result in enumerate(results, 1):
        # Title
        title = highlight_text(result['title'], query_words)
        print(f"\n{Colors.CYAN}{Colors.BOLD}{i}.{Colors.END} {title}")
        
        # URL 
        url = format_url(result['url'])
        print(f"   {Colors.DIM}{url}{Colors.END}")
        
        # Snippet
        if result['snippet']:
            snippet = result['snippet']
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            snippet = highlight_text(snippet, query_words)
            print(f"   {snippet}")
        
        # Score and PageRank
        print(f"   {Colors.DIM}Score: {result['score']:.6f} | PageRank: {result['pagerank']:.6f}{Colors.END}")
    
    print(f"\n{Colors.DIM}{'─' * 80}{Colors.END}\n")


def interactive_mode(engine):
    """Run Bernoulli in interactive mode."""
    print(f"\n{Colors.BOLD}{Colors.ORANGE}{'═' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.ORANGE}  Bernoulli: Search Anything Caltech!{Colors.END}")
    print(f"{Colors.BOLD}{Colors.ORANGE}{'═' * 80}{Colors.END}")
    print(f"{Colors.DIM}Enter search queries (or 'quit'/'exit'/'q' to stop){Colors.END}")
    print(f"{Colors.DIM}Press Ctrl+C to exit{Colors.END}\n")
    
    query_count = 0
    
    while True:
        try:
            query = input(f"{Colors.BOLD}Search:{Colors.END} ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.GREEN}Goodbye!{Colors.END}\n")
                break
            
            # Help command
            if query.lower() in ['help', '?']:
                print(f"\n{Colors.ORANGE}{Colors.BOLD}Help:{Colors.END}")
                print(f"  {Colors.DIM}• Enter a search query to find pages{Colors.END}")
                print(f"  {Colors.DIM}• Type 'quit', 'exit', or 'q' to exit{Colors.END}")
                print(f"  {Colors.DIM}• Type 'help' or '?' for this message{Colors.END}")
                print()
                continue
            
            # Show searching indicator
            print(f"{Colors.DIM}Searching...{Colors.END}", end='', flush=True)
            
            results = engine.search(query)
            query_count += 1
            
            # Clear the "Searching..." message
            print(f"\r{Colors.DIM}{' ' * 20}\r{Colors.END}", end='')
            
            print_results(results, query)
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Goodbye!{Colors.END}\n")
            break
        except EOFError:
            print(f"\n\n{Colors.GREEN}Goodbye!{Colors.END}\n")
            break


def main():
    # Verify that data files exist
    data_dir = 'data'
    
    required_files = ['index.json', 'pagerank.json', 'page_info.json']
    missing = [f for f in required_files if not os.path.exists(f'{data_dir}/{f}')]
    
    if missing:
        print(f"\n{Colors.RED}{Colors.BOLD}Error: Missing required data files{Colors.END}")
        for f in missing:
            print(f"  {Colors.RED}✗{Colors.END} {data_dir}/{f}")
        print(f"\n{Colors.YELLOW}Please run:{Colors.END}")
        print(f"  {Colors.ORANGE}1.{Colors.END} python crawl.py            (to crawl and index the web)")
        print(f"  {Colors.ORANGE}2.{Colors.END} python compute_pagerank.py  (to compute PageRank scores)")
        print()
        sys.exit(1)
    
    # Initialize Bernoulli
    try:
        engine = SearchEngine(data_dir)
    except Exception as e:
        print(f"{Colors.RED}Error loading search engine: {e}{Colors.END}")
        sys.exit(1)
    
    # Check for command-line queries or interactive mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"{Colors.DIM}Searching...{Colors.END}", end='', flush=True)
        results = engine.search(query)
        print(f"\r{Colors.DIM}{' ' * 20}\r{Colors.END}", end='')
        print_results(results, query)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
