import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math

logger = logging.getLogger("dbert.rag.vector_store")

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates the cosine similarity between two floating point vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class VectorStore:
    def __init__(self, workspace_id: str, app_dir: Path = None):
        self.workspace_id = workspace_id
        if app_dir is None:
            self.workspace_dir = Path.home() / ".dbert" / "workspaces" / workspace_id
        else:
            self.workspace_dir = Path(app_dir) / "workspaces" / workspace_id
            
        self.db_dir = self.workspace_dir / "vector_store"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "vectors.db"
        self._init_db()

    def _init_db(self) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_vectors (
                    id TEXT PRIMARY KEY,
                    document_path TEXT NOT NULL,
                    text_content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            conn.commit()
            logger.info(f"Initialized SQLite vector database for workspace {self.workspace_id} at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vector database for workspace {self.workspace_id}: {e}")
            raise e
        finally:
            if conn:
                conn.close()

    def add_chunks(self, document_path: str, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] = None) -> None:
        """Saves text chunks and their corresponding embeddings to the SQLite store."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_path}_{idx}"
                metadata = metadatas[idx] if metadatas else {}
                cursor.execute(
                    "INSERT OR REPLACE INTO document_vectors (id, document_path, text_content, embedding_json, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, document_path, chunk, json.dumps(embedding), json.dumps(metadata))
                )
            conn.commit()
            logger.info(f"Added {len(chunks)} chunks for {document_path} into vector store.")
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            raise e
        finally:
            if conn:
                conn.close()

    def query(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """
        Queries the local SQLite database for the top_k closest chunks using cosine similarity.
        Returns list of: (document_path, text_content, metadata_dict, similarity_score)
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT document_path, text_content, embedding_json, metadata_json FROM document_vectors")
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                doc_path, text_content, embedding_json, metadata_json = row
                embedding = json.loads(embedding_json)
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                similarity = cosine_similarity(query_embedding, embedding)
                results.append((doc_path, text_content, metadata, similarity))
                
            # Sort by similarity descending
            results.sort(key=lambda x: x[3], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def clear(self) -> None:
        """Deletes all vectorized document chunks in this workspace store."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_vectors")
            conn.commit()
            logger.info(f"Cleared vector store for workspace {self.workspace_id}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
        finally:
            if conn:
                conn.close()

    def get_ingested_files(self) -> List[str]:
        """Returns a list of unique document paths that have been ingested into this workspace."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT document_path FROM document_vectors")
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Error querying ingested files: {e}")
            return []
        finally:
            if conn:
                conn.close()
