"""
Core functionality for Bernoulli that combines text matching with PageRank scores.

Ranking Configuration:
All ranking parameters are now in ranking_config.py in the project root.
Edit that file to experiment with different ranking strategies!
"""
import json
import re
import os
import sys
from collections import defaultdict
from .stopwords import is_stop_word

_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ranking_config.py')

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ranking_config", _config_path)
    ranking_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranking_config)
    
    MEANINGFUL_WORD_WEIGHT = ranking_config.MEANINGFUL_WORD_WEIGHT
    STOP_WORD_WEIGHT = ranking_config.STOP_WORD_WEIGHT
    PERFECT_MATCH_BONUS_MIN = ranking_config.PERFECT_MATCH_BONUS_MIN
    ALL_WORDS_MATCH_BONUS_MIN = ranking_config.ALL_WORDS_MATCH_BONUS_MIN
    PARTIAL_MATCH_PENALTY = ranking_config.PARTIAL_MATCH_PENALTY
    TEXT_RELEVANCE_WEIGHT = ranking_config.TEXT_RELEVANCE_WEIGHT
    PAGERANK_WEIGHT = ranking_config.PAGERANK_WEIGHT
except Exception as e:
    # Could not load ranking_config.py, so use default values
    print(f"Warning: Could not load ranking_config.py: {e}", file=sys.stderr)
    print("Using default ranking parameters.", file=sys.stderr)
    MEANINGFUL_WORD_WEIGHT = 3.0
    STOP_WORD_WEIGHT = 1.0
    PERFECT_MATCH_BONUS_MIN = 0.8
    ALL_WORDS_MATCH_BONUS_MIN = 0.7
    PARTIAL_MATCH_PENALTY = 0.5
    TEXT_RELEVANCE_WEIGHT = 0.9
    PAGERANK_WEIGHT = 0.1


class SearchEngine:
    def __init__(self, data_dir='data'):
        """
        Initialize Bernoulli by loading the index, PageRank scores,
        and page metadata.
        """
        self.data_dir = data_dir
        
        with open(f'{data_dir}/index.json', 'r') as f:
            self.index = json.load(f)
        
        with open(f'{data_dir}/pagerank.json', 'r') as f:
            self.pagerank = json.load(f)
        
        with open(f'{data_dir}/page_info.json', 'r') as f:
            self.page_info = json.load(f)
    
    def tokenize(self, query):
        """Tokenize a search query into words, removing stop words."""
        words = re.findall(r'\b[a-z0-9]+\b', query.lower())
        return words
    
    def search(self, query, max_results=10):
        """
        Search for pages matching the query.
        
        Returns results ranked by a combination of:
        - Text relevance (pages with ALL query words ranked highest)
        - PageRank score
        """
        query_words = self.tokenize(query)
        
        if not query_words:
            return []
        
        # Find pages containing each query word
        pages_by_word = {}
        for word in query_words:
            if word in self.index:
                pages_by_word[word] = set(self.index[word])
        
        if not pages_by_word:
            # No words found in index
            return []
        
        # Get all candidate pages (union of all word matches)
        candidate_pages = set()
        for pages in pages_by_word.values():
            candidate_pages.update(pages)
        
        if not candidate_pages:
            return []
        
        # Separate stop words from meaningful words
        stop_words = [w for w in query_words if is_stop_word(w)]
        meaningful_words = [w for w in query_words if not is_stop_word(w)]
        
        # Score each candidate page
        page_scores = {}
        for url in candidate_pages:
            # Count matching words (weight meaningful words more)
            matching_meaningful = sum(1 for word in meaningful_words 
                                    if word in pages_by_word and url in pages_by_word[word])
            matching_stop = sum(1 for word in stop_words 
                              if word in pages_by_word and url in pages_by_word[word])
            matching_total = matching_meaningful + matching_stop
            
            if matching_total == 0:
                continue
            
            # Weight meaningful words more than stop words (configurable)
            weighted_matches = matching_meaningful * MEANINGFUL_WORD_WEIGHT + matching_stop * STOP_WORD_WEIGHT
            max_weighted = len(meaningful_words) * MEANINGFUL_WORD_WEIGHT + len(stop_words) * STOP_WORD_WEIGHT
            
            # Base score: weighted fraction of words matched
            if max_weighted > 0:
                text_score = weighted_matches / max_weighted
            else:
                # All words are stop words - treat normally
                text_score = matching_total / len(query_words)
            
            # Big bonus if page contains ALL words (especially meaningful ones)
            if matching_total == len(query_words):
                if matching_meaningful == len(meaningful_words) and len(meaningful_words) > 0:
                    # Perfect match on meaningful words
                    text_score = PERFECT_MATCH_BONUS_MIN + (1.0 - PERFECT_MATCH_BONUS_MIN) * text_score
                else:
                    # All words matched but some are stop words
                    text_score = ALL_WORDS_MATCH_BONUS_MIN + (1.0 - ALL_WORDS_MATCH_BONUS_MIN) * text_score
            else:
                # Partial matches get lower scores
                text_score = text_score * PARTIAL_MATCH_PENALTY
            
            page_scores[url] = text_score
        
        if not page_scores:
            return []
        
        # Normalize text scores to [0, 1] range
        max_text_score = max(page_scores.values())
        if max_text_score > 0:
            for url in page_scores:
                page_scores[url] /= max_text_score
        
        # Combine text scores with PageRank (weights are configurable above)
        final_scores = {}
        for url, text_score in page_scores.items():
            pr_score = self.pagerank.get(url, 0.0)
            # Normalize PageRank (assuming it's already in reasonable range)
            final_scores[url] = TEXT_RELEVANCE_WEIGHT * text_score + PAGERANK_WEIGHT * pr_score
        
        # Sort by combined score
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top results with metadata
        results = []
        for url, score in sorted_results[:max_results]:
            info = self.page_info.get(url, {})
            results.append({
                'url': url,
                'title': info.get('title', url),
                'snippet': info.get('snippet', ''),
                'score': score,
                'pagerank': self.pagerank.get(url, 0.0)
            })
        
        return results
