import os
import re
import json
import logging
import datetime
from pathlib import Path
import requests
import litellm
from typing import Any

from src.research.web_search import search

logger = logging.getLogger("dbert.research.deep_research_web")

def clean_html_text(html: str) -> str:
    """Strips tags, style sheets, and scripts from raw HTML, returning clean body text."""
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:2000] # Cap to prevent LLM context bloat

def fetch_page_content(url: str) -> str:
    """Fetches a URL and extracts its raw text content. Falls back to empty string on error."""
    logger.info(f"Fetching page content: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DBERTResearch/0.1"}
    try:
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            return clean_html_text(r.text)
    except Exception as e:
        logger.debug(f"Failed to fetch content from {url}: {e}")
    return ""

def run_deep_research(
    query: str,
    workspace_id: str,
    active_model: Any,
    provider_manager: Any,
    config_manager: Any
) -> str:
    """
    Decomposes query, runs multiple web searches, fetches source content,
    and synthesizes a markdown research report with inline citations.
    """
    logger.info(f"Starting web deep research on: '{query}'")
    
    # 1. Query decomposition
    decomp_prompt = (
        f"Generate 3 distinct search query strings to research the following topic: \"{query}\". "
        f"Return them as a JSON list of strings, for example: [\"query1\", \"query2\", \"query3\"]. "
        f"Do not include any conversational text or explanation."
    )
    
    p_info = provider_manager.active_providers.get(active_model.provider, {})
    sub_queries = []
    try:
        if active_model.is_local:
            response = litellm.completion(
                model=f"openai/{active_model.id}",
                messages=[{"role": "user", "content": decomp_prompt}],
                api_base=p_info.get("base_url"),
                api_key="lm-studio",
                timeout=180
            )
        else:
            response = litellm.completion(
                model=active_model.id,
                messages=[{"role": "user", "content": decomp_prompt}],
                api_key=p_info.get("api_key"),
                timeout=180
            )
        text = response.choices[0].message.content.strip()
        # Clean JSON codeblock wrappers if LLM returned them
        if "```" in text:
            text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
        sub_queries = json.loads(text.strip())
    except Exception as e:
        logger.warning(f"Decomposition failed, using original query: {e}")
        sub_queries = [query]
        
    # Guard against invalid JSON parsing fallback
    if not isinstance(sub_queries, list) or len(sub_queries) == 0:
        sub_queries = [query]
        
    logger.info(f"Decomposed sub-queries: {sub_queries}")
    
    # 2. Gather Search Hits & Page Content
    source_index = 1
    citations = {}
    sources_text = []
    
    for sq in sub_queries[:3]:
        search_hits = search(sq)
        for hit in search_hits[:2]: # Extract top 2 hits per query
            url = hit["url"]
            title = hit["title"]
            snippet = hit["snippet"]
            
            # Fetch full page text
            page_text = fetch_page_content(url)
            text_context = page_text if page_text else snippet
            
            citations[source_index] = {"title": title, "url": url}
            sources_text.append(f"[{source_index}] Title: {title}\nURL: {url}\nContent: {text_context}\n")
            source_index += 1
            
    if not sources_text:
        return "Deep research failed: No sources could be fetched or queried."
        
    # 3. Report Synthesis
    synthesis_prompt = f"""
You are a Deep Research Specialist. Please synthesize a comprehensive, cited research report in markdown based on the following gathered sources.
Organize it with headers, bullet points, and inline citations (e.g. [1], [2]) linking to the sources.

Topic to research: {query}

Gathered Sources:
{"---".join(sources_text)}

At the end of the report, list all sources numerically under a "# Sources" header using markdown link notation, matching the inline citation numbers exactly.
"""
    
    logger.info("Synthesizing research report...")
    try:
        if active_model.is_local:
            response = litellm.completion(
                model=f"openai/{active_model.id}",
                messages=[{"role": "user", "content": synthesis_prompt}],
                api_base=p_info.get("base_url"),
                api_key="lm-studio",
                timeout=180
            )
        else:
            response = litellm.completion(
                model=active_model.id,
                messages=[{"role": "user", "content": synthesis_prompt}],
                api_key=p_info.get("api_key"),
                timeout=180
            )
        report_text = response.choices[0].message.content
        
        # Save to workspaces/<workspace_id>/research/
        workspace_dir = Path.home() / ".dbert" / "workspaces" / workspace_id
        if config_manager.app_dir:
            workspace_dir = Path(config_manager.app_dir) / "workspaces" / workspace_id
            
        research_dir = workspace_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = research_dir / f"report_{timestamp}.md"
        report_file.write_text(report_text, encoding="utf-8")
        logger.info(f"Research report saved to {report_file}")
        
        return f"{report_text}\n\n*Saved report to: {report_file}*"
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return f"Failed to synthesize research report: {e}"
