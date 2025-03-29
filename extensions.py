# extensions.py 
"""
Extensions Module.

Initializes and configures Flask extensions used throughout the Resume Analyzer
project, including SQLAlchemy, Migrate, Cache, Compress, and Limiter.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_caching import Cache
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

db = SQLAlchemy()
migrate = Migrate()
cache = Cache()
compress = Compress()

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",  # Use in-memory storage for rate limiting
    default_limits=["200/day"]
)

def init_extensions(app):
    """Initialize all Flask extensions with the provided app instance."""
    # Initialize database and migration support
    db.init_app(app)
    migrate.init_app(app, db)

    # Initialize performance-related extensions
    cache.init_app(app)
    compress.init_app(app)

    # Initialize rate limiter (requires app context)
    limiter.init_app(app)
