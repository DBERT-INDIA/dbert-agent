import sqlite3
import json
import uuid
import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("dbert.session_manager")

@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List] = None
    timestamp: str = ""

@dataclass
class Session:
    id: str
    workspace_id: str
    created_at: str
    name: str
    messages: List[Message] = field(default_factory=list)

@dataclass
class SessionSummary:
    id: str
    workspace_id: str
    created_at: str
    name: str
    last_message: Optional[str] = None

@dataclass
class SessionFilter:
    workspace_id: Optional[str] = None

class SessionManager:
    def __init__(self, db_path: Path = None):
        if db_path is None:
            self.db_path = Path.home() / ".dbert" / "history" / "chat.db"
        else:
            self.db_path = Path(db_path)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages (id) ON DELETE CASCADE
                )
            """)
            conn.commit()
            logger.info(f"Initialized SQLite database at {self.db_path}")
        except Exception as e:
            logger.critical(f"Failed to initialize SQLite session database: {e}")
            raise e
        finally:
            if conn:
                conn.close()

    def create_session(self, workspace_id: str, name: Optional[str] = None) -> Session:
        session_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()
        if not name:
            name = f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (id, workspace_id, created_at, name) VALUES (?, ?, ?, ?)",
                (session_id, workspace_id, created_at, name)
            )
            conn.commit()
            logger.info(f"Created session {session_id} under workspace {workspace_id}")
            return Session(id=session_id, workspace_id=workspace_id, created_at=created_at, name=name, messages=[])
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise e
        finally:
            if conn:
                conn.close()

    def resume_session(self, session_id: str) -> Session:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            session_row = cursor.fetchone()
            if not session_row:
                raise ValueError(f"Session with ID {session_id} not found.")
                
            cursor.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
            msg_rows = cursor.fetchall()
            
            messages = []
            for row in msg_rows:
                tool_calls = None
                if row["tool_calls"]:
                    try:
                        tool_calls = json.loads(row["tool_calls"])
                    except Exception:
                        pass
                messages.append(Message(
                    role=row["role"],
                    content=row["content"],
                    tool_calls=tool_calls,
                    timestamp=row["timestamp"]
                ))
                
            logger.info(f"Resumed session {session_id} with {len(messages)} messages.")
            return Session(
                id=session_row["id"],
                workspace_id=session_row["workspace_id"],
                created_at=session_row["created_at"],
                name=session_row["name"],
                messages=messages
            )
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            raise e
        finally:
            if conn:
                conn.close()

    def list_sessions(self, filter_opts: Optional[SessionFilter] = None) -> List[SessionSummary]:
        query = """
            SELECT s.id, s.workspace_id, s.created_at, s.name,
                   (SELECT content FROM messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1) as last_msg
            FROM sessions s
        """
        params = []
        if filter_opts and filter_opts.workspace_id:
            query += " WHERE s.workspace_id = ?"
            params.append(filter_opts.workspace_id)
            
        query += " ORDER BY s.created_at DESC"
        
        conn = None
        try:
            summaries = []
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            for row in rows:
                summaries.append(SessionSummary(
                    id=row["id"],
                    workspace_id=row["workspace_id"],
                    created_at=row["created_at"],
                    name=row["name"],
                    last_message=row["last_msg"]
                ))
            return summaries
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def append_message(self, session_id: str, role: str, content: str, tool_calls: Optional[List] = None, embedding: Optional[List[float]] = None) -> None:
        timestamp = datetime.datetime.now().isoformat()
        tool_calls_str = json.dumps(tool_calls) if tool_calls else None
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, tool_calls_str, timestamp)
            )
            msg_id = cursor.lastrowid
            
            if embedding:
                cursor.execute(
                    "INSERT INTO message_embeddings (message_id, session_id, role, content, embedding_json) VALUES (?, ?, ?, ?, ?)",
                    (msg_id, session_id, role, content, json.dumps(embedding))
                )
                
            conn.commit()
            logger.debug(f"Appended message ({role}) to session {session_id}")
        except Exception as e:
            logger.error(f"Failed to append message to session {session_id}: {e}")
            raise e
        finally:
            if conn:
                conn.close()
