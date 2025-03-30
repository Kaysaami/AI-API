# wsgi_minimal.py
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print diagnostic information
logger.info(f"Starting minimal Flask application")
port = int(os.environ.get("PORT", 10000))
logger.info(f"Using port: {port}")

# Create a minimal Flask app
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        "status": "online",
        "message": "Minimal API is running. Full application features disabled due to memory constraints."
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "environment": os.environ.get("RENDER_ENV", "unknown")
    })

# For direct execution
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
