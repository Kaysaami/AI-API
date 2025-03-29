# models/user.py
"""
User model for authentication and user management in the Resume Analyzer project.
"""

from datetime import datetime, timedelta
import uuid
import jwt
from flask import current_app
from extensions import db
from services.security import hash_password, verify_password

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)

    # Relationships 
    uploads = db.relationship('Upload', back_populates='user', lazy='dynamic', cascade='all, delete-orphan')
    
    analyses = db.relationship('AnalysisResult', back_populates='user', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.email}>'

    def set_password(self, password):
        """Set the user's password hash using the security module."""
        self.password_hash = hash_password(password)

    def verify_password(self, password):
        """Verify a password against the stored hash."""
        return verify_password(self.password_hash, password)

    def generate_auth_token(self, expires_in=86400):
        """Generate a JWT authentication token valid for specified seconds (default 24h)."""
        payload = {
            'user_id': self.id,
            'email': self.email,
            'is_admin': self.is_admin,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        return jwt.encode(
            payload,
            current_app.config.get('SECRET_KEY', 'dev-key'),
            algorithm='HS256'
        )

    def generate_reset_token(self, expires_in=3600):
        """Generate password reset token valid for specified seconds (default 1h)."""
        self.reset_token = str(uuid.uuid4())
        self.reset_token_expires = datetime.utcnow() + timedelta(seconds=expires_in)
        return self.reset_token

    def clear_reset_token(self):
        """Clear password reset token."""
        self.reset_token = None
        self.reset_token_expires = None

    @staticmethod
    def verify_auth_token(token):
        """Verify JWT token and return User object if valid."""
        try:
            payload = jwt.decode(
                token,
                current_app.config.get('SECRET_KEY', 'dev-key'),
                algorithms=['HS256']
            )
            return User.query.get(payload['user_id'])
        except jwt.ExpiredSignatureError:
            return None  # Token expired
        except jwt.InvalidTokenError:
            return None  # Invalid token
        except Exception:
            return None  # Any other error