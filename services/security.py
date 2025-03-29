#services/security.py
from datetime import datetime
import re
from flask import g, current_app
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from extensions import db

def hash_password(password):
    """Hash a password using Werkzeug security"""
    return generate_password_hash(password)

def verify_password(password_hash, password):
    """Verify a password against a hash using Werkzeug security"""
    return check_password_hash(password_hash, password)

def validate_password_strength(password):
    """
    Validate password strength:
    - At least 8 characters
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one number
    - Contains at least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter."
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number."
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character."
    
    return True, "Password is strong."

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    return False

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
    
    # Check if this is the first user - make them admin
    is_first_user = User.query.count() == 0
    
    # Create new user
    user = User(
        email=email,
        first_name=first_name,
        last_name=last_name,
        is_admin=is_first_user  # First user becomes admin
    )
    user.password = password  # This calls the password.setter method
    
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
    
    if not user.verify_password(password):
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
    user.password = new_password
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
    
def validate_password_complexity(password):
    """Alias for validate_password_strength for API consistency"""
    is_valid, _ = validate_password_strength(password)
    return is_valid

def get_password_rules():
    """Return password requirements for client-side validation"""
    return {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_number": True,
        "require_special": True,
        "special_chars": "!@#$%^&*(),.?\":{}|<>"
    }