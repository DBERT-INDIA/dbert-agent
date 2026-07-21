import sqlite3
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List
from src.rag.vector_store import cosine_similarity

logger = logging.getLogger("dbert.memory.history_search")

@dataclass
class HistoryMatch:
    session_id: str
    role: str
    content: str
    similarity: float

def semantic_search_history(
    db_path: Path,
    query_embedding: List[float],
    workspace_id: str,
    top_k: int = 3
) -> List[HistoryMatch]:
    """
    Searches past chat messages belonging to this workspace,
    compares embeddings using cosine similarity, and returns the top K hits.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Retrieve message embeddings for sessions belonging to this workspace
        cursor.execute("""
            SELECT me.session_id, me.role, me.content, me.embedding_json
            FROM message_embeddings me
            JOIN sessions s ON me.session_id = s.id
            WHERE s.workspace_id = ?
        """, (workspace_id,))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            session_id, role, content, embedding_json = row
            embedding = json.loads(embedding_json)
            
            similarity = cosine_similarity(query_embedding, embedding)
            results.append(HistoryMatch(
                session_id=session_id,
                role=role,
                content=content,
                similarity=similarity
            ))
            
        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]
    except Exception as e:
        logger.error(f"Failed to query chat history embeddings: {e}")
        return []
    finally:
        if conn:
            conn.close()
