# routes/auth_routes.py
"""
Authentication Routes - Endpoints for user login, registration, and token verification.
"""

from flask import Blueprint, request, jsonify, g, current_app
from functools import wraps
from datetime import datetime
import jwt

from extensions import limiter, db
from models.user import User
from services.security import validate_password_complexity, get_password_rules
from routes.auth_utils import token_required

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """
    User login endpoint.
    Expects JSON payload with 'email' and 'password'.
    Returns a JWT token on successful authentication.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request data"}), 400
        
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    
    user = User.query.filter_by(email=email).first()

    # Debug: Check if user exists and what's stored in the database
    if not user:
        current_app.logger.error(f"Login failed: No user found with email {email}")
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Debug: Check if password verification works
    password_valid = user.verify_password(password)
    current_app.logger.info(f"Password verification result: {password_valid}")
    
    if not password_valid:
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Update last login time
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Generate token
    token = user.generate_auth_token(
        expires_in=current_app.config.get('JWT_EXPIRATION', 86400)
    )

    return jsonify({
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "is_admin": user.is_admin
        }
    })
@auth_bp.route('/register', methods=['POST'])
@limiter.limit("2 per hour")
def register():
    """
    User registration endpoint.
    Expects JSON payload with 'email', 'password', 'first_name', and 'last_name'.
    """
    data = request.get_json()

    # Validate required fields
    if not data:
        return jsonify({"error": "Missing request data"}), 400
        
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # Check for existing user
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already exists"}), 409
    
    # Validate password complexity
    if not validate_password_complexity(password):
        return jsonify({
            "error": "Password does not meet complexity requirements",
            "password_rules": get_password_rules()
        }), 400

    # Create and save new user
    user = User(
        email=email,
        first_name=first_name,
        last_name=last_name
    )
    
    # Make the first user an admin
    user_count = User.query.count()
    if user_count == 0:
        user.is_admin = True
    
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "User created successfully",
        "user_id": user.id
    }), 201

@auth_bp.route('/me', methods=['GET'])
@token_required
def get_user_profile():
    """Get current user profile information."""
    user = g.current_user
    
    return jsonify({
        "id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    })

@auth_bp.route('/password-reset-request', methods=['POST'])
@limiter.limit("3 per hour")
def request_password_reset():
    """Request a password reset token."""
    data = request.get_json()
    
    if not data or 'email' not in data:
        return jsonify({"error": "Email is required"}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    # Always return success even if user doesn't exist (security best practice)
    if not user:
        return jsonify({
            "message": "If your email is registered, you will receive a password reset link"
        }), 200
    
    # Generate reset token
    token = user.generate_reset_token()
    db.session.commit()
    
    # In a real app, you would send an email with the reset link here
    # For now, just return the token (not secure for production)
    
    return jsonify({
        "message": "Password reset token generated",
        "token": token  # In production, don't return this directly to the user
    }), 200

@auth_bp.route('/password-reset', methods=['POST'])
@limiter.limit("3 per hour")
def reset_password():
    """Reset password using a token."""
    data = request.get_json()
    
    if not data or 'token' not in data or 'new_password' not in data:
        return jsonify({"error": "Token and new password are required"}), 400
    
    user = User.query.filter_by(reset_token=data['token']).first()
    
    if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
        return jsonify({"error": "Invalid or expired token"}), 400
    
    # Validate password complexity
    if not validate_password_complexity(data['new_password']):
        return jsonify({
            "error": "Password does not meet complexity requirements",
            "password_rules": get_password_rules()
        }), 400
    
    # Update password and clear token
    user.set_password(data['new_password'])
    user.clear_reset_token()
    db.session.commit()
    
    return jsonify({"message": "Password reset successful"}), 200

@auth_bp.route('/change-password', methods=['POST'])
@token_required
@limiter.limit("3 per hour")
def change_password():
    """Change user password (requires authentication)."""
    data = request.get_json()
    user = g.current_user
    
    if not data or 'current_password' not in data or 'new_password' not in data:
        return jsonify({"error": "Current and new passwords are required"}), 400
    
    # Verify current password
    if not user.verify_password(data['current_password']):
        return jsonify({"error": "Current password is incorrect"}), 400
    
    # Validate new password complexity
    if not validate_password_complexity(data['new_password']):
        return jsonify({
            "error": "Password does not meet complexity requirements",
            "password_rules": get_password_rules()
        }), 400
    
    # Update password
    user.set_password(data['new_password'])
    db.session.commit()
    
    return jsonify({"message": "Password changed successfully"}), 200