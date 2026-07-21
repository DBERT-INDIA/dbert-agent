import requests
import re
import logging
from urllib.parse import unquote, quote_plus

logger = logging.getLogger("dbert.research.web_search")

def search(query: str) -> list[dict]:
    """
    Queries DuckDuckGo HTML search page and scrapes standard organic results.
    Returns: list of {"title", "url", "snippet"}
    """
    logger.info(f"Initiating Web Search for query: '{query}'")
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            logger.warning(f"DuckDuckGo returned status code {r.status_code}")
            return []
            
        html = r.text
        
        # Scrape titles and links
        # Match pattern: <h2 class="result__title"><a class="result__snippet" href="LINK">TITLE</a></h2>
        # or similar. Let's do a loose matching search on result__snippet inside titles
        matches = re.findall(r'<h2 class="result__title">.*?href="([^"]+)".*?>(.*?)</a>', html, re.DOTALL)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        
        results = []
        for i, (link, title) in enumerate(matches[:5]):
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_link = unquote(link)
            
            # Skip duckduckgo redirect strings or ads if they leak into results
            if clean_link.startswith("//duckduckgo.com/y.js") or "y.js?" in clean_link:
                continue
                
            # If the URL is relative, prepend protocol
            if clean_link.startswith("//"):
                clean_link = "https:" + clean_link
                
            snippet_text = ""
            if i < len(snippets):
                snippet_text = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                
            results.append({
                "title": clean_title,
                "url": clean_link,
                "snippet": snippet_text
            })
            
        if not results:
            logger.warning("No standard results extracted from DuckDuckGo search HTML structure.")
            return [{
                "title": "Search Error", 
                "url": url, 
                "snippet": "The web search engine failed to return parseable results. Consider rephrasing or searching a specific site."
            }]
            
        logger.info(f"Found {len(results)} search result items.")
        return results
    except Exception as e:
        logger.error(f"Failed to query and scrape web search: {e}")
        return []
