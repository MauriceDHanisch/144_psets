# **Bernoulli: Search Anything Caltech!**

A mini search engine for the Caltech domain, built for CMS/CS/Ec/EE 144.

## Setup

Make sure you're using Python 3. The project uses standard library only (no external dependencies needed) and standard Python packages.

## Running the Search Engine

### Step 1: Crawl and Index the Web

This crawls the Caltech domain and automatically builds the search index:

```bash
python crawl.py
```

**Options:**
- `python crawl.py [seed_url] [max_pages]`
- Example: `python crawl.py http://www.caltech.edu/ 2500`

**What it does:**
- Crawls pages starting from the seed URL
- Extracts links to build the web graph
- Indexes page content for search
- Saves to `data/graph.json`, `data/index.json`, and `data/page_info.json`

**Performance Notes:**
- After crawling ~5,000 nodes, we found reasonably good search performance
- Be aware of potential memory constraints on your device that might limit how large of a web graph you can crawl, index, and search over
- Consider your available RAM when choosing `max_pages` - larger graphs require more memory

### Step 2: Implement and Compute PageRank Scores

**This is your main implementation task!** You need to implement the `compute_pagerank()` function in `src/pagerank.py`.

After implementing PageRank, compute PageRank scores for ranking:

```bash
python compute_pagerank.py
```

**What it does:**
- Loads the web graph from `data/graph.json`
- Calls your `compute_pagerank()` implementation to compute PageRank scores
- Saves scores to `data/pagerank.json`
- Shows top 10 pages by PageRank

**Implementation Notes:**
- Implement `compute_pagerank()` in `src/pagerank.py`
- The function signature and docstring are provided - you just need to implement the algorithm

### Step 3: Search!

Now you can search:

```bash
# Interactive mode (Recommended)
python search.py

# Single query
python search.py "your search query"
```

**Example searches:**
```bash
python search.py "CMS/CS/Ec/EE 144"
python search.py "Thomas Rosenbaum"
python search.py "Admissions Office"
python search.py "Diya Kumar"
python search.py "[Your Name]"
python search.py "[Your Friend's Name]"
```

## Workflow Summary

```bash
# 1. Crawl and Index
python crawl.py

# 2. Compute PageRank
python compute_pagerank.py

# 3. Search
python search.py
```

## File Structure

```
bernoulli/
├── crawl.py              # Run the crawler
├── compute_pagerank.py   # Compute PageRank scores
├── search.py             # Search CLI interface
├── ranking_config.py     # Ranking parameters - edit this to tune search quality!
├── data/                 # Generated data files
│   ├── graph.json        # Web graph (links) - Generated after first crawl!
│   ├── index.json        # Search index (words -> pages) - Generated after first crawl!
│   ├── page_info.json    # Page titles and snippets - Generated after first crawl!
│   ├── pagerank.json     # PageRank scores - Generated after computing PageRank scores!
│   └── stopwords.txt     # Stop words list
└── src/                  # Core package
    ├── crawler.py        # Web crawler
    ├── fetcher.py        # HTML fetching
    ├── indexer.py        # Text extraction & indexing
    ├── pagerank.py       # PageRank algorithm (implement this!)
    ├── search_util.py    # SearchEngine class
    └── stopwords.py      # Stop word utilities
```

## Notes

- **PageRank**: You need to implement `compute_pagerank()` in `src/pagerank.py`
- **Crawling**: The crawler indexes pages during crawling (efficient!)
- **Search Ranking**: Results are ranked by a combination of text relevance and PageRank
  - Default weights: 90% text relevance, 10% PageRank
  - **All ranking parameters are in `ranking_config.py`** - edit this file to experiment with different ranking strategies!
  - No need to modify `src/search_util.py` - just change values in `ranking_config.py` and restart the search engine

## Extra Credit Opportunity

**Up to 10 points** for creative improvements to the search engine! This is your chance to have some fun and explore. Ideas include:

- **Custom Ranking Algorithms**: Implement your own ranking algorithm
- **AI-Enabled Search**: Build your own Perplexity clone (you will need to supply your own API key or run infra locally)
- **User Interface**: Build a web interface or GUI for the search engine
- **Query Expansion**: Implement synonym expansion, query suggestions, or auto-complete
- **Advanced Features**: Add filters (date, domain), result clustering, or personalized search
- **Performance Optimization**: Optimize indexing, caching, or search speed
- **Your Own Ideas**: Surprise us! Be creative and document what you built

To submit extra credit work, create a brief write-up explaining what you implemented and how it works.

## Troubleshooting

**Missing data files?**
- Make sure you've run `crawl.py` first, then `compute_pagerank.py`

**No search results?**
- Check that `data/index.json` and `data/pagerank.json` exist
- Try re-crawling to rebuild the index

**PageRank not working?**
- Make sure you've implemented `compute_pagerank()` in `src/pagerank.py`
- Check that `compute_pagerank.py` runs without errors - it will show you the top 10 pages by PageRank
- Try running `python src/pagerank.py` to test your implementation on simple networks