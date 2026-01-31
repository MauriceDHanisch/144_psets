"""
A web crawler for the Caltech domain that explores pages while building an
induced subgraph and an inverted index to later be used by the search engine.
"""

from . import fetcher
from .indexer import extract_text_from_html, tokenize
import json
import re
import html
import time
from urllib.parse import urlparse, urldefrag
from collections import deque, defaultdict

class Crawler:
    def __init__(self, seed_url="http://www.caltech.edu/", max_pages=2000):
        self.max_pages = max_pages
        self.seed = self.normalize_url(seed_url)
        
        # Use BFS traversal
        self.queue = deque([self.seed])
        self.visited = set()
        self.bad_urls = set()
        
        self.in_links = {}
        self.out_links = {}
        
        # Store the complete graph structure as an adjacency list
        self.graph = {}
        
        # Search index
        self.index = defaultdict(list)  # word -> list of URLs
        self.page_info = {}  # url -> {title, snippet}
        
        # Extensions to skip
        self.skip_extensions = {
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.exe', '.dmg', '.pkg', '.deb', '.rpm',
            '.css', '.js', '.xml', '.rss', '.json'
        }
    
    def normalize_url(self, url):
        """
        Normalize URL by removing fragments, standardizing format, and
        converting http:// to https:// to avoid duplicate entries.
        """
        url, _ = urldefrag(url)
        url = url.rstrip('/')
        
        if url.startswith('http://'):
            url = 'https://' + url[7:]
        elif not url.startswith('https://'):
            url = 'https://' + url
            
        return url
    
    def _index_page(self, url, html_content):
        """
        Index a page's content for search.
        Extracts text, tokenizes, and builds the inverted index.
        Returns the number of unique words indexed.
        """
        try:
            # Extract text from HTML
            text = extract_text_from_html(html_content)
            words = tokenize(text)
            
            # Extract title for display
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = title_match.group(1)
                title = re.sub(r'<[^>]+>', '', title)
                title = html.unescape(title)
                # Normalize whitespace: replace all whitespace sequences with single space
                title = re.sub(r'\s+', ' ', title).strip()
            else:
                title = url
            title = title[:100]
            
            # Build snippet from body text
            # Try to get a clean snippet from paragraphs or body content
            snippet = text[:300].strip()
            snippet = re.sub(r'\s+', ' ', snippet)
            if len(snippet) > 200:
                snippet = snippet[:200].rsplit(' ', 1)[0] + '...'
            
            # Store page information
            self.page_info[url] = {
                'title': title,
                'snippet': snippet
            }
            
            # Add words to inverted index
            unique_words = set(words)
            words_added = 0
            for word in unique_words:
                if url not in self.index[word]:
                    self.index[word].append(url)
                    words_added += 1
            
            return len(unique_words)
        except Exception as e:
            # If indexing fails, just skip it for now
            return 0
    
    def is_valid_caltech_url(self, url):
        """Check if URL is within Caltech domain and should be crawled"""
        # Must be within the caltech.edu domain
        if not bool(re.match(r"^https?://([a-zA-Z0-9-]+\.)*caltech\.edu(/.*)?$", url)):
            return False
        
        # Skip potentially problematic extensions
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in self.skip_extensions):
            return False
        
        return True
    
    def crawl(self, delay=0.1):
        """Main crawling loop with politeness delay"""
        print(f"Starting crawl from {self.seed}")
        print(f"Target: {self.max_pages} pages\n")
        
        while self.queue and len(self.visited) < self.max_pages:
            # Use BFS
            url = self.queue.popleft()
            
            # Normalize URL before processing
            url = self.normalize_url(url)
            
            # Skip if already visited or known to be bad
            if url in self.visited or url in self.bad_urls:
                continue
            
            # Politeness delay to avoid DoS
            time.sleep(delay)
            
            # Fetch page content
            real_url, html_content = fetcher.fetch_html_page(url)
            
            if html_content is not None:
                normalized_url = self.normalize_url(real_url)
                
                if normalized_url in self.visited:
                    continue
                
                self.visited.add(normalized_url)
                
                # Extract links from the page using our HTMLParser
                parser = fetcher.MyHTMLParser()
                parser.urls = []
                parser.feed(html_content)
                parser.close()
                links = parser.get_links(real_url)
                
                normalized_links = []
                for link in links:
                    link = self.normalize_url(link)
                    if self.is_valid_caltech_url(link):
                        normalized_links.append(link)
                        
                        # Add to queue if not yet visited (using normalized URL)
                        if link not in self.visited and \
                           link not in self.bad_urls and \
                           link not in self.queue:
                            self.queue.append(link)
                
                self.graph[normalized_url] = normalized_links
                
                words_indexed = self._index_page(normalized_url, html_content)
                
                print(f"{len(self.visited):5d}/{len(self.visited) + len(self.queue):5d} "
                      f"{normalized_url} (out: {len(normalized_links)}, words: {words_indexed})")
            else:
                self.bad_urls.add(url)
                print(f"{len(self.visited):5d}/{len(self.visited) + len(self.queue):5d} "
                      f"{url} (FAILED)")
        
        # Build induced subgraph
        self._build_induced_subgraph()
        
        print(f"\nCrawl complete!")
        print(f"URLs visited: {len(self.visited)}")
        print(f"URLs in queue: {len(self.queue)}")
        print(f"Bad URLs: {len(self.bad_urls)}")
    
    def _build_induced_subgraph(self):
        """
        Build induced subgraph containing only edges between visited nodes
        """
        print("\nBuilding induced subgraph...")
        
        # Initialize in-degree and out-degree counts
        self.in_links = {url: 0 for url in self.visited}
        self.out_links = {url: 0 for url in self.visited}
        
        # Filter graph to only include edges between visited nodes
        induced_graph = {}
        
        for source_url in self.graph:
            # Only consider source URLs that were visited
            if source_url not in self.visited:
                continue
            
            # Filter target URLs to only include visited nodes (i.e. only include edges between visited nodes)
            valid_targets = [
                target for target in self.graph[source_url] 
                if target in self.visited
            ]
            
            induced_graph[source_url] = valid_targets
            self.out_links[source_url] = len(valid_targets)
            
            for target in valid_targets:
                self.in_links[target] += 1
        
        self.graph = induced_graph
        
        print(f"Induced subgraph: {len(self.visited)} nodes, "
              f"{sum(self.out_links.values())} edges")
    
    def save_results(self, data_dir='data'):
        """Save crawl results to JSON files, including search index"""
        import os
        os.makedirs(data_dir, exist_ok=True)
        
        # Save our web graph
        with open(f'{data_dir}/graph.json', 'w') as f:
            json.dump(self.graph, f, indent=2)
        
        # Save search index
        index_dict = {word: urls for word, urls in self.index.items()}
        with open(f'{data_dir}/index.json', 'w') as f:
            json.dump(index_dict, f, indent=2)
        
        # Save page info (titles, snippets) to display in search results
        with open(f'{data_dir}/page_info.json', 'w') as f:
            json.dump(self.page_info, f, indent=2)
        
        print(f"\nResults saved to {data_dir}/")
        print(f"  - Graph: {len(self.graph)} pages")
        print(f"  - Index: {len(index_dict)} unique words")
        print(f"  - Page info: {len(self.page_info)} pages")