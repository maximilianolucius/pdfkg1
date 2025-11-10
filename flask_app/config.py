"""Configuration for Flask application."""
import os
from datetime import timedelta


class Config:
    """Flask application configuration."""

    # Basic Flask config
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    # Session configuration
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)

    # File upload configuration
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = 'data/input'

    # Storage backend
    STORAGE_BACKEND = os.getenv('STORAGE_BACKEND', 'arango')

    # Database configuration
    ARANGO_HOST = os.getenv('ARANGO_HOST', 'localhost')
    ARANGO_PORT = os.getenv('ARANGO_PORT', '8529')
    ARANGO_USER = os.getenv('ARANGO_USER', 'root')
    ARANGO_PASSWORD = os.getenv('ARANGO_PASSWORD', '')
    ARANGO_DB = os.getenv('ARANGO_DB', 'pdfkg')

    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')

    # LLM API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')

    # Default embedding model
    DEFAULT_EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    DEFAULT_EMBED_DIM = os.getenv('DEFAULT_EMBED_DIM', '384')

    # Processing defaults
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_TOP_K = 5
