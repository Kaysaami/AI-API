# services/security_db.py
from datetime import datetime
import jwt
from flask import g, current_app
from extensions import db
from resume_matcher_services.security import (
    hash_password,
    verify_password, 
    validate_password_strength,
    validate_email,
    validate_password_complexity
)

def register_user(email, password, first_name, last_name):
    """Register a new user"""
    # Import User locally to avoid circular import
    from models.user import User
    
    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return None, "Email already registered."
    
    # Validate email format
    if not validate_email(email):
        return None, "Invalid email format."
    
    # Validate password strength
    is_valid, message = validate_password_strength(password)
    if not is_valid:
        return None, message
    
    # Check if this is the first non-test user - make them admin
    is_first_non_test_user = User.query.filter(User.email != 'test@example.com').count() == 0
    
    # Create new user
    user = User(
        email=email,
        first_name=first_name,
        last_name=last_name,
        is_admin=is_first_non_test_user  # First non-test user becomes admin
    )
    user.password_hash = hash_password(password)
    
    # Add to database
    db.session.add(user)
    db.session.commit()
    
    return user, "User registered successfully."

def authenticate_user(email, password):
    """Authenticate a user"""
    # Import User locally to avoid circular import
    from models.user import User
    
    user = User.query.filter_by(email=email).first()
    
    if not user or not user.is_active:
        return None, "Invalid email or password."
    
    if not verify_password(user.password_hash, password):
        return None, "Invalid email or password."
    
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    return user, "Authentication successful."

def request_password_reset(email):
    """Request a password reset"""
    # Import User locally to avoid circular import
    from models.user import User
    
    user = User.query.filter_by(email=email).first()
    
    if not user:
        # Don't reveal that the user doesn't exist for security reasons
        return None, "If your email is registered, you will receive a password reset link."
    
    # Generate reset token
    token = user.generate_reset_token()
    db.session.commit()
    
    return user, token

def reset_password(token, new_password):
    """Reset password using a token"""
    # Import User locally to avoid circular import
    from models.user import User
    
    user = User.query.filter_by(reset_token=token).first()
    
    if not user or user.reset_token_expires < datetime.utcnow():
        return False, "Invalid or expired token."
    
    # Validate password strength
    is_valid, message = validate_password_strength(new_password)
    if not is_valid:
        return False, message
    
    # Update password and clear token
    user.password_hash = hash_password(new_password)
    user.clear_reset_token()
    db.session.commit()
    
    return True, "Password reset successfully."

def decode_auth_token(token):
    """Decode an authentication token"""
    try:
        payload = jwt.decode(
            token,
            current_app.config['SECRET_KEY'],
            algorithms=['HS256']
        )
        return payload, None
    except jwt.ExpiredSignatureError:
        return None, "Token expired. Please log in again."
    except jwt.InvalidTokenError:
        return None, "Invalid token. Please log in again."
