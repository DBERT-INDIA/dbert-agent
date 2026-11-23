"""Central configuration, overridable through environment variables."""
import os


class Config:
    """Default settings for Aivara."""

    DATABASE = os.environ.get("AIVARA_DB", "aivara.db")
    API_KEY = os.environ.get("AIVARA_API_KEY", "dev-key-change-me")
    MAX_STEPS = 20
    MAX_RETRIES = 3
    RATE_LIMIT = 60  # requests per minute per client
    MAX_CONTENT_LENGTH = 64 * 1024
