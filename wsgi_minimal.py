# wsgi_minimal.py
import os
import sys
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")

# Create a minimal Flask app instead of importing your full app
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from Render! Minimal app is working."
