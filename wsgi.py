# wsgi.py
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print diagnostic information
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")

# Properly handle the PORT environment variable
port = int(os.environ.get("PORT", 10000))
logger.info(f"Using port: {port}")

try:
    # Import your app factory function
    from app import create_app
    
    # Create the Flask application using your factory function
    # Use 'production' config for Render deployment
    logger.info("Creating application with production config...")
    app = create_app('production')
    
    logger.info("Application created successfully!")
    
    # Optional: If you want to run this file directly (not through gunicorn)
    if __name__ == "__main__":
        logger.info(f"Starting application on port {port}...")
        app.run(host="0.0.0.0", port=port)

except Exception as e:
    logger.error(f"Failed to create application: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    
    # Create a minimal fallback app for diagnostics
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def error_page():
        return f"Application startup error: {str(e)}"
