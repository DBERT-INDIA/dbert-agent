import logging
import sqlite3
from typing import Any
import litellm
from src.rag.vector_store import VectorStore

logger = logging.getLogger("dbert.rag.summarizer")

def summarize_document(
    workspace_id: str,
    document_path: str,
    active_model: Any,
    provider_manager: Any,
    app_dir: Any = None
) -> str:
    """
    Retrieves document chunks from the VectorStore, sends them to the active model,
    and returns a summary of the document.
    """
    logger.info(f"Summarizing document: {document_path}")
    
    # Retrieve chunks from SQLite
    vector_store = VectorStore(workspace_id, app_dir=app_dir)
    
    conn = None
    text_content = ""
    try:
        conn = sqlite3.connect(vector_store.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT text_content FROM document_vectors WHERE document_path = ? ORDER BY id ASC",
            (document_path,)
        )
        rows = cursor.fetchall()
        # Concatenate text up to 8000 characters to prevent overloading context windows
        text_content = "\n".join([row[0] for row in rows])
        if len(text_content) > 8000:
            text_content = text_content[:8000] + "\n...[TRUNCATED]..."
    except Exception as e:
        logger.error(f"Failed to read document chunks for summarization: {e}")
        return f"Error reading document: {e}"
    finally:
        if conn:
            conn.close()
            
    if not text_content.strip():
        return "No text available to summarize."
        
    prompt = f"Please provide a concise summary of the following document contents, highlighting key points:\n\n{text_content}"
    
    p_info = provider_manager.active_providers.get(active_model.provider, {})
    try:
        if active_model.is_local:
            response = litellm.completion(
                model=f"openai/{active_model.id}",
                messages=[{"role": "user", "content": prompt}],
                api_base=p_info.get("base_url"),
                api_key="lm-studio"
            )
        else:
            response = litellm.completion(
                model=active_model.id,
                messages=[{"role": "user", "content": prompt}],
                api_key=p_info.get("api_key")
            )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Summarization failed: {e}")
        return f"Failed to generate summary: {e}"
