# routes/auth_utils.py
"""
Authentication utility functions and decorators to be used across route files.
"""

from flask import request, jsonify, g, current_app
from functools import wraps
import jwt
from models.user import User

def token_required(f):
    """Decorator to protect routes with JWT token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({"error": "Authentication token is missing"}), 401
        
        try:
            # Decode the token
            payload = jwt.decode(
                token, 
                current_app.config.get('SECRET_KEY', 'dev-key'),
                algorithms=['HS256']
            )
            
            # Find the user
            current_user = User.query.get(payload['user_id'])
            
            if not current_user or not current_user.is_active:
                return jsonify({"error": "Invalid or inactive user"}), 401
            
            # Store user in flask g object for the current request
            g.current_user = current_user
            
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        except Exception as e:
            current_app.logger.error(f"Token validation error: {str(e)}")
            return jsonify({"error": "Token validation failed"}), 401
            
        return f(*args, **kwargs)
    
    return decorated

def admin_required(f):
    """Decorator to protect routes with JWT token authentication and admin check."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({"error": "Authentication token is missing"}), 401
        
        try:
            # Decode the token
            payload = jwt.decode(
                token, 
                current_app.config.get('SECRET_KEY', 'dev-key'),
                algorithms=['HS256']
            )
            
            # Find the user
            current_user = User.query.get(payload['user_id'])
            
            if not current_user or not current_user.is_active:
                return jsonify({"error": "Invalid or inactive user"}), 401
                
            # Check if user is an admin
            if not current_user.is_admin:
                return jsonify({"error": "Admin privileges required"}), 403
            
            # Store user in flask g object for the current request
            g.current_user = current_user
            
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        except Exception as e:
            current_app.logger.error(f"Token validation error: {str(e)}")
            return jsonify({"error": "Token validation failed"}), 401
            
        return f(*args, **kwargs)
    
    return decorated