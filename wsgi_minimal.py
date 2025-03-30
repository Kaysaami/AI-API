# wsgi_minimal.py
import os
import sys
import logging
import traceback

# Configure more detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Print diagnostic information
logger.info(f"Python version: {sys.version}")
logger.info(f"Starting minimal Flask application")
port = int(os.environ.get("PORT", 10000))
logger.info(f"Using port: {port}")

try:
    # Create a minimal Flask app
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    @app.before_request
    def log_request_info():
        logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr}")
        return None

    @app.after_request
    def log_response_info(response):
        logger.debug(f"Response: {response.status_code}")
        return response

    @app.route('/')
    def hello():
        logger.info("Root endpoint called")
        return jsonify({
            "status": "online",
            "message": "Minimal API is running. Full application features disabled due to memory constraints."
        })

    @app.route('/health')
    def health_check():
        logger.info("Health check endpoint called")
        return jsonify({
            "status": "healthy",
            "environment": os.environ.get("RENDER_ENV", "unknown")
        })

    # Catch-all for 404 errors
    @app.errorhandler(404)
    def not_found(e):
        logger.warning(f"404 error: {request.path}")
        return jsonify({"error": "not found"}), 404

    # For direct execution
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=port)

except Exception as e:
    logger.error(f"Error during application startup: {e}")
    logger.error(traceback.format_exc())
    raise
