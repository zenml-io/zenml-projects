# Exa Python SDK Implementation Cheat Sheet

## Overview

Exa is a neural search engine designed for AI applications. Unlike traditional keyword-based search, Exa uses embeddings to understand semantic meaning, making it ideal for sophisticated search queries.

## Installation & Setup

```python
pip install exa-py
```

```python
from exa_py import Exa
import os

# Initialize client
exa = Exa(os.getenv("EXA_API_KEY"))
```

## Core Methods Comparison

### Search Methods

| **Method** | **Purpose** | **Returns** |
|------------|-------------|-------------|
| `search()` | Basic search, returns links only | List of Result objects with URLs, titles, scores |
| `search_and_contents()` | Search + content retrieval in one call | Results with full text/highlights |
| `get_contents()` | Get content for specific document IDs | Content for provided IDs |
| `find_similar()` | Find pages similar to a URL | Similar pages |
| `find_similar_and_contents()` | Find similar + get content | Similar pages with content |

### Key Differences from Tavily

1. **Neural vs Keyword Search**: Exa defaults to neural search but supports keyword search via `type="keyword"`
2. **Content Integration**: Exa can retrieve full content in the same API call
3. **Semantic Similarity**: `find_similar()` functionality for finding related content
4. **Structured Response**: More predictable response structure

## Basic Search Implementation

### Simple Search (Links Only)

```python
def exa_search_basic(query: str, num_results: int = 3) -> Dict[str, Any]:
    try:
        response = exa.search(
            query=query,
            num_results=num_results,
            type="auto"  # "auto", "neural", or "keyword"
        )
        
        return {
            "query": query,
            "results": [
                {
                    "url": result.url,
                    "title": result.title,
                    "score": result.score,
                    "published_date": result.published_date,
                    "author": result.author,
                    "id": result.id  # Temporary document ID
                }
                for result in response.results
            ]
        }
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}
```

### Search with Content (Primary Method)

```python
def exa_search_with_content(
    query: str,
    num_results: int = 3,
    max_characters: int = 20000,
    include_highlights: bool = False
) -> Dict[str, Any]:
    try:
        # Configure content options
        text_options = {"max_characters": max_characters}
        
        kwargs = {
            "query": query,
            "num_results": num_results,
            "text": text_options,
            "type": "auto"
        }
        
        # Add highlights if requested
        if include_highlights:
            kwargs["highlights"] = {
                "highlights_per_url": 2,
                "num_sentences": 3
            }
        
        response = exa.search_and_contents(**kwargs)
        
        return {
            "query": query,
            "results": [
                {
                    "url": result.url,
                    "title": result.title,
                    "content": result.text,  # Full text content
                    "highlights": getattr(result, 'highlights', []),
                    "score": result.score,
                    "published_date": result.published_date,
                    "author": result.author,
                    "id": result.id
                }
                for result in response.results
            ]
        }
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}
```

## Advanced Search Parameters

### Date Filtering

```python
response = exa.search_and_contents(
    query="AI research",
    start_published_date="2024-01-01",
    end_published_date="2024-12-31",
    start_crawl_date="2024-01-01",  # When Exa crawled the content
    end_crawl_date="2024-12-31"
)
```

### Domain Filtering

```python
response = exa.search_and_contents(
    query="machine learning",
    include_domains=["arxiv.org", "scholar.google.com"],
    exclude_domains=["reddit.com", "twitter.com"]
)
```

### Search Types

```python
# Neural search (default, semantic understanding)
neural_results = exa.search(query, type="neural")

# Keyword search (Google-style)
keyword_results = exa.search(query, type="keyword") 

# Auto (Exa chooses best approach)
auto_results = exa.search(query, type="auto")
```

## Content Options Deep Dive

### Text Content Options

```python
text_options = {
    "max_characters": 5000,        # Limit content length
    "include_html_tags": True      # Keep HTML formatting
}

response = exa.search_and_contents(
    query="AI developments",
    text=text_options
)
```

### Highlights Options

```python
highlights_options = {
    "highlights_per_url": 3,       # Number of highlights per result
    "num_sentences": 2,            # Sentences per highlight
    "query": "custom highlight query"  # Override search query for highlights
}

response = exa.search_and_contents(
    query="machine learning",
    highlights=highlights_options
)
```

### Combined Content Retrieval

```python
# Get both full text and highlights
response = exa.search_and_contents(
    query="quantum computing",
    text={"max_characters": 10000},
    highlights={
        "highlights_per_url": 2,
        "num_sentences": 1
    }
)

# Results will have both .text and .highlights attributes
for result in response.results:
    print(f"Title: {result.title}")
    print(f"Full text: {result.text[:200]}...")
    print(f"Highlights: {result.highlights}")
```

## Response Structure Analysis

### Basic Result Object

```python
class Result:
    url: str                    # The webpage URL
    id: str                     # Temporary document ID for get_contents()
    title: Optional[str]        # Page title
    score: Optional[float]      # Relevance score (0-1)
    published_date: Optional[str]  # Estimated publication date
    author: Optional[str]       # Content author if available
```

### Result with Content

```python
class ResultWithText(Result):
    text: str                   # Full page content (when text=True)

class ResultWithHighlights(Result):
    highlights: List[str]       # Key excerpts
    highlight_scores: List[float]  # Relevance scores for highlights

class ResultWithTextAndHighlights(Result):
    text: str
    highlights: List[str]
    highlight_scores: List[float]
```

## Error Handling & Retry Logic

```python
def exa_search_with_retry(
    query: str,
    max_retries: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """Search with retry logic similar to your Tavily implementation"""
    
    # Alternative query strategies
    query_variants = [
        query,
        f'"{query}"',  # Exact phrase
        f"research about {query}",
        f"article on {query}"
    ]
    
    for attempt in range(max_retries + 1):
        try:
            current_query = query_variants[min(attempt, len(query_variants) - 1)]
            
            response = exa.search_and_contents(
                query=current_query,
                **kwargs
            )
            
            # Check if we got meaningful content
            content_results = sum(
                1 for r in response.results 
                if hasattr(r, 'text') and r.text.strip()
            )
            
            if content_results > 0:
                return {
                    "query": current_query,
                    "results": [
                        {
                            "url": r.url,
                            "content": getattr(r, 'text', ''),
                            "title": r.title or '',
                            "snippet": ' '.join(getattr(r, 'highlights', [])[:1]),
                            "score": r.score,
                            "published_date": r.published_date,
                            "author": r.author
                        }
                        for r in response.results
                    ]
                }
            
        except Exception as e:
            if attempt == max_retries:
                return {"query": query, "results": [], "error": str(e)}
            continue
    
    return {"query": query, "results": []}
```

## Mapping to Your SearchResult Model

Based on your Pydantic `SearchResult` model, here's the mapping:

```python
def convert_exa_to_search_result(exa_result) -> SearchResult:
    """Convert Exa result to your SearchResult format"""
    
    # Get the best available content
    content = ""
    if hasattr(exa_result, 'text') and exa_result.text:
        content = exa_result.text
    elif hasattr(exa_result, 'highlights') and exa_result.highlights:
        content = f"Title: {exa_result.title}\n\nHighlights:\n" + "\n".join(exa_result.highlights)
    elif exa_result.title:
        content = f"Title: {exa_result.title}"
    
    # Create snippet from highlights or title
    snippet = ""
    if hasattr(exa_result, 'highlights') and exa_result.highlights:
        snippet = exa_result.highlights[0]
    elif exa_result.title:
        snippet = exa_result.title
    
    return SearchResult(
        url=exa_result.url,
        content=content,
        title=exa_result.title or "",
        snippet=snippet
    )
```

## Additional Features

### Find Similar Content

```python
def find_similar_content(url: str, num_results: int = 3):
    """Find content similar to a given URL"""
    response = exa.find_similar_and_contents(
        url=url,
        num_results=num_results,
        text=True,
        exclude_source_domain=True  # Don't include same domain
    )
    return response.results
```

### Get Content by IDs

```python
def get_content_by_ids(document_ids: List[str]):
    """Retrieve full content for specific document IDs"""
    response = exa.get_contents(
        ids=document_ids,
        text={"max_characters": 10000}
    )
    return response.results
```

### Answer API (Tavily Alternative)

```python
def exa_answer_query(query: str, include_full_text: bool = False):
    """Get direct answer to question (similar to Tavily's answer feature)"""
    response = exa.answer(
        query=query,
        text=include_full_text  # Include full text of citations
    )
    
    return {
        "answer": response.answer,
        "citations": [
            {
                "url": citation.url,
                "title": citation.title,
                "text": getattr(citation, 'text', '') if include_full_text else '',
                "published_date": citation.published_date
            }
            for citation in response.citations
        ]
    }
```

## Configuration Toggle Implementation

```python
class SearchConfig:
    TAVILY = "tavily"
    EXA = "exa"

def unified_search(
    query: str,
    provider: str = SearchConfig.EXA,
    **kwargs
) -> List[SearchResult]:
    """Unified search interface supporting both providers"""
    
    if provider == SearchConfig.EXA:
        exa_results = exa_search_with_content(query, **kwargs)
        return [
            convert_exa_to_search_result(result) 
            for result in exa_results.get("results", [])
        ]
    elif provider == SearchConfig.TAVILY:
        tavily_results = tavily_search(query, **kwargs)
        return extract_search_results(tavily_results)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

## Performance & Cost Considerations

### Pricing Structure
- **Neural Search**: $0.005 for 1-25 results, $0.025 for 26-100 results
- **Keyword Search**: $0.0025 for 1-100 results  
- **Content**: $0.001 per page for text/highlights/summary
- **Live Crawling**: Automatic fallback for uncached content

### Optimization Tips

1. **Use `search_and_contents()`** instead of separate `search()` + `get_contents()` calls
2. **Set appropriate `max_characters`** to control content length and costs
3. **Choose search type wisely**: neural for semantic queries, keyword for exact matches
4. **Use domain filtering** to focus on high-quality sources
5. **Implement caching** for repeated queries

## Environment Variables

```bash
EXA_API_KEY=your_exa_api_key_here
```

## Testing Queries

```python
# Test neural search capabilities
test_queries = [
    "fascinating article about machine learning",
    "comprehensive guide to Python optimization",
    "latest research in quantum computing",
    "how to implement search functionality"
]

for query in test_queries:
    results = exa_search_with_content(query, num_results=2)
    print(f"Query: {query}")
    print(f"Results: {len(results['results'])}")
    print("---")
```

This cheat sheet provides everything you need to implement Exa as a configurable
alternative to Tavily, maintaining compatibility with your existing
`SearchResult` model while leveraging Exa's advanced neural search capabilities.
