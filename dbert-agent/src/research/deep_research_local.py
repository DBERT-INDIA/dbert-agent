import os
import re
import json
import logging
import datetime
from pathlib import Path
import litellm
from typing import Any

from src.rag.ingest import get_embeddings
from src.rag.vector_store import VectorStore
from src.memory.history_search import semantic_search_history

logger = logging.getLogger("dbert.research.deep_research_local")

def run_local_research(
    query: str,
    workspace_id: str,
    active_model: Any,
    provider_manager: Any,
    config_manager: Any
) -> str:
    """
    Runs query decomposition and searches local document indexes and chat history,
    synthesizing a cited report from local workspace assets.
    """
    logger.info(f"Starting local deep research on: '{query}'")
    
    # 1. Query decomposition
    decomp_prompt = (
        f"Generate 3 distinct search query strings to research the following topic in our local document indexes: \"{query}\". "
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
        if "```" in text:
            text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
        sub_queries = json.loads(text.strip())
    except Exception as e:
        logger.warning(f"Decomposition failed, using original query: {e}")
        sub_queries = [query]
        
    if not isinstance(sub_queries, list) or len(sub_queries) == 0:
        sub_queries = [query]
        
    logger.info(f"Local sub-queries: {sub_queries}")
    
    # 2. Gather Document Chunks & Chat history
    source_index = 1
    sources_text = []
    
    # Instantiate stores
    app_dir = config_manager.app_dir
    vs = VectorStore(workspace_id, app_dir=app_dir)
    
    from src.core.session_manager import SessionManager
    sm = SessionManager(app_dir=app_dir)
    
    for sq in sub_queries[:3]:
        # Generate embedding
        try:
            emb = get_embeddings([sq], active_model, provider_manager)[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding for '{sq}': {e}")
            continue
            
        # Query Document Chunks
        doc_matches = vs.query(emb, top_k=2)
        for doc_path, chunk_text, meta, score in doc_matches:
            if score > 0.3:
                sources_text.append(f"[{source_index}] Local Document: {Path(doc_path).name} (Score: {score:.2f})\nContent: {chunk_text}\n")
                source_index += 1
                
        # Query Chat History
        history_matches = semantic_search_history(sm.db_path, emb, workspace_id, top_k=2)
        for match in history_matches:
            if match.similarity > 0.35:
                sources_text.append(f"[{source_index}] Past Conversation: Session {match.session_id[:8]} ({match.role}) (Score: {match.similarity:.2f})\nContent: {match.content}\n")
                source_index += 1
                
    if not sources_text:
        return "Local deep research completed: No matching local document chunks or history transcripts found."
        
    # 3. Report Synthesis
    synthesis_prompt = f"""
You are a Local Asset Research Specialist. Please synthesize a comprehensive research report in markdown based on the following local document chunks and past session dialogue history.
Organize it with headers, bullet points, and inline citations (e.g. [1], [2]) linking to the local sources.

Topic to research: {query}

Gathered Sources:
{"---".join(sources_text)}

At the end of the report, list all sources numerically under a "# Sources" header (identifying the document name or session ID).
"""
    
    logger.info("Synthesizing local research report...")
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
        if app_dir:
            workspace_dir = Path(app_dir) / "workspaces" / workspace_id
            
        research_dir = workspace_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = research_dir / f"report_local_{timestamp}.md"
        report_file.write_text(report_text, encoding="utf-8")
        logger.info(f"Local research report saved to {report_file}")
        
        return f"{report_text}\n\n*Saved local report to: {report_file}*"
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return f"Failed to synthesize local research report: {e}"
