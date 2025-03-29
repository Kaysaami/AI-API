# error_handlers.py
"""
Error Handlers Module.

Defines custom error handlers for the Flask application to ensure consistent
error responses across the API.
"""

from flask import jsonify

def register_error_handlers(app):
    """Register error handlers with the Flask application."""
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle 400 Bad Request errors."""
        return jsonify({
            "error": "Bad request",
            "message": str(error.description) if hasattr(error, 'description') else "Invalid request parameters"
        }), 400
    
    @app.errorhandler(401)
    def handle_unauthorized(error):
        """Handle 401 Unauthorized errors."""
        return jsonify({
            "error": "Unauthorized",
            "message": "Authentication required"
        }), 401
    
    @app.errorhandler(403)
    def handle_forbidden(error):
        """Handle 403 Forbidden errors."""
        return jsonify({
            "error": "Forbidden",
            "message": "You don't have permission to access this resource"
        }), 403
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 Not Found errors."""
        return jsonify({
            "error": "Not found",
            "message": "The requested resource was not found"
        }), 404
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        return jsonify({
            "error": "Method not allowed",
            "message": "The method is not allowed for the requested URL"
        }), 405
    
    @app.errorhandler(422)
    def handle_unprocessable_entity(error):
        """Handle 422 Unprocessable Entity errors."""
        return jsonify({
            "error": "Unprocessable entity",
            "message": "The request was well-formed but was unable to be followed due to semantic errors"
        }), 422
    
    @app.errorhandler(429)
    def handle_rate_limit_exceeded(error):
        """Handle 429 Too Many Requests errors."""
        return jsonify({
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later."
        }), 429
    
    @app.errorhandler(500)
    def handle_internal_server_error(error):
        """Handle 500 Internal Server Error."""
        app.logger.error(f"Internal server error: {str(error)}")
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
    
    # Custom exception handler for SQLAlchemy errors
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle uncaught exceptions."""
        app.logger.error(f"Uncaught exception: {str(error)}")
        return jsonify({
            "error": "Server error",
            "message": "An unexpected error occurred"
        }), 500