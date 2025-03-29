# config.py

"""
Configuration Module for Resume Analyzer.

Defines configuration settings for different environments (development,
testing, production) using a dataclass.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import secrets

# Load environment variables from .env file
load_dotenv()

# Now you can access OPENAI_API_KEY via os.getenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("Loaded OPENAI_API_KEY:", OPENAI_API_KEY) 

@dataclass
class Config:
    SQLALCHEMY_DATABASE_URI: str
    SQLALCHEMY_TRACK_MODIFICATIONS: bool
    MAX_CONTENT_LENGTH: int  # Maximum allowed content length in bytes
    UPLOAD_FOLDER: Path
    ALLOWED_EXTENSIONS: set
    SECRET_KEY: str
    JWT_EXPIRATION: int  # Token expiration time in seconds

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return configuration settings for different environments."""
        # Generate a secret key if not already set in environment
        secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
        
        return {
            'development': {
                'SQLALCHEMY_DATABASE_URI': 'sqlite:///resume_analyzer.db',
                'SQLALCHEMY_TRACK_MODIFICATIONS': False,
                'MAX_CONTENT_LENGTH': 5 * 1024 * 1024,  # 5MB
                'UPLOAD_FOLDER': Path('uploads'),
                'ALLOWED_EXTENSIONS': {'pdf', 'docx'},
                'SECRET_KEY': secret_key,
                'JWT_EXPIRATION': 86400,  # 24 hours in seconds
                'DEBUG': True
            },
            'testing': {
                'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
                'SQLALCHEMY_TRACK_MODIFICATIONS': False,
                'MAX_CONTENT_LENGTH': 5 * 1024 * 1024,  # 5MB
                'UPLOAD_FOLDER': Path('test_uploads'),
                'ALLOWED_EXTENSIONS': {'pdf', 'docx'},
                'SECRET_KEY': 'test-key',
                'JWT_EXPIRATION': 3600,  # 1 hour in seconds
                'TESTING': True
            },
            'production': {
                'SQLALCHEMY_DATABASE_URI': os.getenv('DATABASE_URI', 'postgresql://user:pass@localhost/resumedb'),
                'SQLALCHEMY_TRACK_MODIFICATIONS': False,
                'MAX_CONTENT_LENGTH': 5 * 1024 * 1024,  # 5MB
                'UPLOAD_FOLDER': Path('/var/www/uploads'),
                'ALLOWED_EXTENSIONS': {'pdf', 'docx'},
                'SECRET_KEY': secret_key,
                'JWT_EXPIRATION': 86400,  # 24 hours in seconds
                'DEBUG': False
            }
        }

import openai
import os

def setup_openai():
    # Retrieve the key from environment variables
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Warning: OPENAI_API_KEY not set.")
    else:
        print("OpenAI API key loaded successfully!")