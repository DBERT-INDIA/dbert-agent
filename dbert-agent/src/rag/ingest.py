import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import litellm

from src.rag.pdf_parser import extract_text_and_tables
from src.rag.vector_store import VectorStore

logger = logging.getLogger("dbert.rag.ingest")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunks a block of text using a sliding window algorithm.
    Splits at whitespace boundaries where possible to avoid clipping words.
    """
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # If we aren't at the end of the text, try to back off to a whitespace
        if end < text_len:
            # Look back up to 50 chars for a space or newline, but not before start
            lookback_start = max(start, end - 50)
            lookback = text[lookback_start:end]
            space_idx = max(lookback.rfind(" "), lookback.rfind("\n"))
            if space_idx != -1:
                end = lookback_start + space_idx
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        if end == text_len:
            break
            
        # Guarantee meaningful forward progress
        start = max(start + max(1, chunk_size // 4), end - overlap)
            
    return chunks

def get_embeddings(texts: List[str], active_model: Any, provider_manager: Any) -> List[List[float]]:
    """
    Retrieves embeddings for a list of texts using the active provider or a local fallback.
    """
    p_info = provider_manager.active_providers.get(active_model.provider, {})
    has_embedding = (active_model.provider != "anthropic")
    
    if has_embedding:
        try:
            if active_model.is_local:
                base_url = p_info.get("base_url")
                response = litellm.embedding(
                    model=f"openai/{active_model.id}",
                    input=texts,
                    api_base=base_url,
                    api_key="lm-studio"
                )
            else:
                api_key = p_info.get("api_key")
                if active_model.provider == "openai":
                    model_name = "openai/text-embedding-3-small"
                elif active_model.provider == "gemini":
                    model_name = "gemini/text-embedding-004"
                else:
                    model_name = "openai/text-embedding-3-small"
                    
                response = litellm.embedding(
                    model=model_name,
                    input=texts,
                    api_key=api_key
                )
            return [item["embedding"] for item in response["data"]]
        except Exception as e:
            logger.warning(f"Failed to fetch embedding via provider {active_model.provider}: {e}. Trying local fallback...")
            
    # Local fallback
    if "lmstudio-local" in provider_manager.active_providers:
        local_info = provider_manager.active_providers["lmstudio-local"]
        try:
            available = provider_manager.list_available_models()
            local_models = [m for m in available if m.provider == "lmstudio-local"]
            
            # Prioritize models with 'embed' in their ID
            local_models.sort(key=lambda m: 0 if "embed" in m.id.lower() else 1)
            
            for model in local_models:
                try:
                    response = litellm.embedding(
                        model=f"openai/{model.id}",
                        input=texts,
                        api_base=local_info.get("base_url"),
                        api_key="lm-studio"
                    )
                    return [item["embedding"] for item in response["data"]]
                except Exception as ex_model:
                    logger.warning(f"Local model {model.id} failed to embed: {ex_model}")
        except Exception as ex:
            logger.error(f"Local embedding fallback failed: {ex}")
            
    raise Exception("No active embedding provider could resolve the embedding request.")

def ingest_file(
    path: str,
    workspace_id: str,
    provider_manager: Any,
    active_model: Any,
    app_dir: Path = None,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Reads a local file, chunks it, retrieves embeddings, and saves it into the VectorStore.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found at {path}")
        
    ext = file_path.suffix.lower()
    
    if ext == ".pdf":
        text = extract_text_and_tables(str(file_path))
    elif ext in [".txt", ".md", ".py", ".js", ".json", ".csv", ".yaml", ".yml"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {path}: {e}")
            raise e
    else:
        raise ValueError(f"Unsupported file type for ingestion: {ext}")
        
    if not text.strip():
        return {
            "success": False,
            "document_path": str(file_path),
            "chunks_count": 0,
            "error": "Document contains no extractable text."
        }
        
    # Chunk
    chunks = chunk_text(text)
    
    # Batch embeddings retrieval
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = get_embeddings(batch_chunks, active_model, provider_manager)
        embeddings.extend(batch_embeddings)
        
    # Save to Vector Store
    vector_store = VectorStore(workspace_id, app_dir=app_dir)
    
    metadatas = [{"source": file_path.name, "chunk_index": idx} for idx in range(len(chunks))]
    vector_store.add_chunks(str(file_path), chunks, embeddings, metadatas)
    
    return {
        "success": True,
        "document_path": str(file_path),
        "chunks_count": len(chunks),
        "file_size_bytes": file_path.stat().st_size
    }
