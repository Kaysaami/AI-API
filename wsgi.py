# wsgi.py
import os
import sys
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")

from app import create_app

app = create_app('production')  # Adjust config name as needed