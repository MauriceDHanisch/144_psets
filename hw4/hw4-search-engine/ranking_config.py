"""
Ranking Configuration for Bernoulli Search Engine

This file contains all the tunable parameters for the search ranking algorithm.
Feel free to experiment with these values to improve search quality!

To modify the ranking behavior, simply edit the values below and restart the search engine.

Here are some suggestions for exploring the parameter space:
1. More PageRank influence:
    TEXT_RELEVANCE_WEIGHT = 0.7
    PAGERANK_WEIGHT = 0.3
2. Stricter matching (penalize partial matches more):
    PARTIAL_MATCH_PENALTY = 0.3
3. Stronger preference for perfect matches:
    PERFECT_MATCH_BONUS_MIN = 0.9
4. Less distinction between meaningful and stop words:
    MEANINGFUL_WORD_WEIGHT = 2.0
    STOP_WORD_WEIGHT = 1.0
"""

# WORD WEIGHTING
# How much to weight meaningful (non-stop) words vs stop words
# Higher values mean meaningful words have more impact on ranking
MEANINGFUL_WORD_WEIGHT = 3.0
STOP_WORD_WEIGHT = 1.0

# MATCH TYPE BONUSES
# Minimum score when a page matches ALL meaningful words in the query
# Range: 0.0 to 1.0 (higher = stronger preference for perfect matches)
PERFECT_MATCH_BONUS_MIN = 0.8

# Minimum score when a page matches ALL words (including stop words)
# Range: 0.0 to 1.0
ALL_WORDS_MATCH_BONUS_MIN = 0.7

# Multiplier for pages that only match SOME query words (partial matches)
# Range: 0.0 to 1.0 (lower = stronger penalty for partial matches)
PARTIAL_MATCH_PENALTY = 0.5

# FINAL RANKING WEIGHTS
# How much to weight text relevance vs PageRank in final ranking
# Note that these need not sum to 1.0
TEXT_RELEVANCE_WEIGHT = 0.9
PAGERANK_WEIGHT = 0.1