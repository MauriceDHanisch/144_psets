"""
Search queries often contain stop words that are not useful for search. This module provides
utilities to filter out stop words from queries, allowing the search engine to focus on more
relevant keywords.

Note:
- Stop words are still indexed (e.g., so "rock and roll" matches properly), but are weighted
  lower in ranking.
- The list of stop words is loaded from a stopwords.txt file in the data directory. You can
  edit this file to add or remove words as needed or to explore the impact on search results.
"""
import os

STOPWORDS_FILE = 'stopwords.txt'

# Load stop words from file
_STOP_WORDS = None

def _load_stop_words():
    """Load stop words from stopwords.txt file."""
    global _STOP_WORDS
    if _STOP_WORDS is not None:
        return _STOP_WORDS
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    # src/stopwords.py -> go up one level to project root, then into data/
    project_root = os.path.dirname(module_dir)
    data_dir = os.path.join(project_root, 'data')
    stopwords_file = os.path.join(data_dir, STOPWORDS_FILE)
    
    _STOP_WORDS = set()
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    _STOP_WORDS.add(line.lower())
    except FileNotFoundError:
        _STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    return _STOP_WORDS


def is_stop_word(word):
    """Check if a word is a stop word."""
    return word.lower() in _load_stop_words()


def filter_stop_words(words):
    """Filter out stop words from a list, but keep them if it's the only word."""
    non_stop = [w for w in words if not is_stop_word(w)]
    return non_stop if non_stop else words
