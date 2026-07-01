"""Session manager for Claude CLI backend.

Maps conversation fingerprints to Claude CLI session IDs, enabling the CLI to
maintain context across requests when using the same session. Sessions are
stored in memory with a TTL and optionally persisted to disk.
"""

import hashlib
import json
import os
import time
import uuid
from typing import Dict, Optional, Any, List
from src.core.logging import logger


SESSION_TTL_SECONDS = 24 * 60 * 60  # 24 hours
SESSION_FILE = os.path.join(
    os.environ.get("HOME", "/tmp"), ".claude-code-proxy-sessions.json"
)


class SessionManager:
    """Manages conversation-to-session mappings for the Claude CLI backend."""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._loaded = False

    def _load(self):
        """Load sessions from disk (lazy, once)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._sessions = data
            logger.info(f"SessionManager: loaded {len(self._sessions)} sessions from disk")
        except (FileNotFoundError, json.JSONDecodeError):
            self._sessions = {}

    def _save(self):
        """Persist sessions to disk (fire-and-forget)."""
        try:
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(self._sessions, f, indent=2)
        except Exception as e:
            logger.warning(f"SessionManager: failed to save sessions: {e}")

    @staticmethod
    def _fingerprint_messages(messages: List[Any]) -> str:
        """Create a stable hash of the conversation prefix (all but last message).

        Two requests with the same conversation history (excluding the latest
        message) map to the same session, allowing the CLI to reuse context.
        """
        if len(messages) <= 1:
            return ""

        # Hash everything except the last message
        prefix_parts = []
        for msg in messages[:-1]:
            # Handle both pydantic models and dicts
            if hasattr(msg, "model_dump"):
                data = msg.model_dump()
            elif isinstance(msg, dict):
                data = msg
            else:
                data = {"role": str(getattr(msg, "role", "")),
                        "content": str(getattr(msg, "content", ""))}
            prefix_parts.append(f"{data.get('role', '')}:{data.get('content', '')}")

        joined = "\n".join(prefix_parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def get_or_create(self, messages: List[Any], model: str) -> Optional[str]:
        """Get an existing CLI session ID for a conversation, or create one.

        Returns a session UUID string, or None for single-message conversations
        (no session needed).
        """
        self._load()

        fingerprint = self._fingerprint_messages(messages)
        if not fingerprint:
            return None  # Single message — no session needed

        now = time.time()

        # Check for existing session
        existing = self._sessions.get(fingerprint)
        if existing:
            existing["last_used_at"] = now
            existing["model"] = model
            logger.debug(f"SessionManager: reusing session {existing['session_id'][:8]}...")
            return existing["session_id"]

        # Create new session
        session_id = str(uuid.uuid4())
        self._sessions[fingerprint] = {
            "session_id": session_id,
            "created_at": now,
            "last_used_at": now,
            "model": model,
        }
        logger.info(f"SessionManager: created session {session_id[:8]}... for model={model}")
        self._save()
        return session_id

    def cleanup(self) -> int:
        """Remove expired sessions. Returns the number removed."""
        self._load()
        cutoff = time.time() - SESSION_TTL_SECONDS
        expired = [k for k, v in self._sessions.items() if v.get("last_used_at", 0) < cutoff]
        for key in expired:
            del self._sessions[key]
        if expired:
            logger.info(f"SessionManager: cleaned up {len(expired)} expired sessions")
            self._save()
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._sessions)


# Singleton instance
session_manager = SessionManager()