"""
Index web pages for full-text search and store the index and page information.

This module fetches page content and builds an inverted index
mapping words to the pages that contain them.

It also stores the page information for display later.
"""

import json
import re
import html
from urllib.parse import urlparse
from collections import defaultdict
from . import fetcher


def extract_text_from_html(html_content):
    """
    Extract visible text from HTML content, focusing on main body content.
    Tries to extract meaningful text from semantic HTML elements.
    """
    if not html_content:
        return ""
    
    # Remove script, style, and other non-content elements
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<svg[^>]*>.*?</svg>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Extract title for weighting
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1) if title_match else ""
    if title:
        title = re.sub(r'<[^>]+>', '', title)
        title = html.unescape(title)
        title = re.sub(r'\s+', ' ', title).strip()
    
    # Try to extract body content
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.IGNORECASE | re.DOTALL)
    if body_match:
        body_content = body_match.group(1)
    else:
        # Otherwise, use entire content
        body_content = html_content
    
    # Remove common navigation/header/footer elements
    # These often contain repetitive, non-meaningful text
    body_content = re.sub(r'<nav[^>]*>.*?</nav>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    body_content = re.sub(r'<header[^>]*>.*?</header>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    body_content = re.sub(r'<footer[^>]*>.*?</footer>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    body_content = re.sub(r'<aside[^>]*>.*?</aside>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Extract text from semantic content elements
    text_parts = []
    
    # Headings
    for level in [1, 2, 3, 4, 5, 6]:
        headings = re.findall(rf'<h{level}[^>]*>(.*?)</h{level}>', body_content, re.IGNORECASE | re.DOTALL)
        for heading in headings:
            heading_text = re.sub(r'<[^>]+>', '', heading).strip()
            # Decode HTML entities
            heading_text = html.unescape(heading_text)
            if heading_text:
                # Repeat headings for extra weight
                text_parts.append(heading_text)
                text_parts.append(heading_text)
    
    # Main content areas
    main_content = re.findall(r'<main[^>]*>(.*?)</main>', body_content, re.IGNORECASE | re.DOTALL)
    article_content = re.findall(r'<article[^>]*>(.*?)</article>', body_content, re.IGNORECASE | re.DOTALL)
    section_content = re.findall(r'<section[^>]*>(.*?)</section>', body_content, re.IGNORECASE | re.DOTALL)
    
    # Paragraphs
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', body_content, re.IGNORECASE | re.DOTALL)
    
    # Lists
    list_items = re.findall(r'<li[^>]*>(.*?)</li>', body_content, re.IGNORECASE | re.DOTALL)
    
    # Collect all content
    for content in main_content + article_content + section_content:
        content_text = re.sub(r'<[^>]+>', ' ', content).strip()
        content_text = html.unescape(content_text)
        if content_text:
            text_parts.append(content_text)
    
    for para in paragraphs:
        para_text = re.sub(r'<[^>]+>', ' ', para).strip()
        para_text = html.unescape(para_text)
        if para_text:
            text_parts.append(para_text)
    
    for item in list_items:
        item_text = re.sub(r'<[^>]+>', ' ', item).strip()
        item_text = html.unescape(item_text)
        if item_text:
            text_parts.append(item_text)
    
    # If we didn't find semantic elements, fall back to extracting all text
    if not text_parts:
        # Remove remaining HTML tags and get text
        body_text = re.sub(r'<[^>]+>', ' ', body_content).strip()
        body_text = html.unescape(body_text)
        if body_text:
            text_parts.append(body_text)
    
    # Combine all text parts
    body_text = ' '.join(text_parts)
    
    # Clean up whitespace
    body_text = re.sub(r'\s+', ' ', body_text).strip()
    
    # Combine title with body text
    if title:
        # Title appears multiple times for extra weight in search
        full_text = f"{title} {title} {title} {body_text}"
    else:
        full_text = body_text
    
    return full_text


def tokenize(text):
    """
    Split text into words (tokens).
    Convert to lowercase and extract alphanumeric sequences.
    """
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return words


def build_index(graph, data_dir='data'):
    """
    Build an inverted index from the web graph.
    
    For each page in the graph, fetch its content and index the words.
    Saves the index to data/index.json.
    """
    print("Building search index...")
    
    # Inverted index: word -> list of URLs containing that word
    index = defaultdict(list)
    
    # Store page titles and snippets for display later
    page_info = {}
    
    urls = list(graph.keys())
    total = len(urls)
    
    for i, url in enumerate(urls, 1):
        if i % 50 == 0:
            print(f"  Indexed {i}/{total} pages...")
        
        try:
            # Fetch page content
            real_url, content = fetcher.fetch_html_page(url)
            
            if content:
                # Extract and tokenize text
                text = extract_text_from_html(content)
                words = tokenize(text)
                
                # Extract title for display
                title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
                if title_match:
                    title = title_match.group(1)
                    title = re.sub(r'<[^>]+>', '', title)  # Clean up any nested tags
                    title = html.unescape(title)  # Decode HTML entities
                    # Normalize whitespace: replace all whitespace sequences with single space
                    title = re.sub(r'\s+', ' ', title).strip()
                else:
                    title = url
                title = title[:100]  # Limit title length
                
                # Build snippet from body text (first meaningful 200 chars)
                snippet = text[:300].strip()
                snippet = re.sub(r'\s+', ' ', snippet)  # Normalize whitespace
                # Truncate at word boundary if needed
                if len(snippet) > 200:
                    snippet = snippet[:200].rsplit(' ', 1)[0] + '...'
                
                page_info[url] = {
                    'title': title,
                    'snippet': snippet
                }
                
                # Add words to index
                unique_words = set(words)
                for word in unique_words:
                    if word not in index or url not in index[word]:
                        index[word].append(url)
        
        except Exception as e:
            # Skip pages that fail to fetch
            continue
    
    # Convert defaultdict to regular dict for JSON serialization
    index_dict = {word: urls for word, urls in index.items()}
    
    # Save index
    index_path = f'{data_dir}/index.json'
    with open(index_path, 'w') as f:
        json.dump(index_dict, f, indent=2)
    
    # Save page info
    info_path = f'{data_dir}/page_info.json'
    with open(info_path, 'w') as f:
        json.dump(page_info, f, indent=2)
    
    print(f"\nIndex built successfully!")
    print(f"  Total pages indexed: {len(page_info)}")
    print(f"  Total unique words: {len(index_dict)}")
    print(f"  Index saved to {index_path}")
    print(f"  Page info saved to {info_path}")
    
    return index_dict, page_info