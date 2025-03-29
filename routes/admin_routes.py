# routes/admin_routes.py
"""
Admin Routes - Protected endpoints for administrative functions.
Only accessible to users with admin privileges.
"""

from flask import Blueprint, request, jsonify, g, current_app
from extensions import db, limiter
from models.user import User
from datetime import datetime
from functools import wraps
import jwt
from routes.auth_utils import admin_required

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# User management endpoints (admin only)
@admin_bp.route('/users', methods=['GET'])
@admin_required
@limiter.limit("20 per minute")
def list_users():
    """List all users (admin only)."""
    users = User.query.all()
    
    user_list = [{
        "id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "is_active": user.is_active,
        "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    } for user in users]
    
    return jsonify({"users": user_list})

@admin_bp.route('/users/<user_id>', methods=['GET'])
@admin_required
def get_user(user_id):
    """Get a specific user's details (admin only)."""
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    user_data = {
        "id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "is_active": user.is_active,
        "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }
    
    return jsonify(user_data)

@admin_bp.route('/users/<user_id>/toggle-status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Enable or disable a user account (admin only)."""
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    # Don't allow admins to disable their own account
    if user.id == g.current_user.id:
        return jsonify({"error": "Cannot modify your own account status"}), 400
        
    user.is_active = not user.is_active
    db.session.commit()
    
    return jsonify({
        "message": f"User {user.email} {'activated' if user.is_active else 'deactivated'} successfully",
        "is_active": user.is_active
    })

@admin_bp.route('/users/<user_id>/toggle-admin', methods=['POST'])
@admin_required
def toggle_admin_status(user_id):
    """Toggle a user's admin status (admin only)."""
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    # Don't allow admins to remove their own admin privileges
    if user.id == g.current_user.id:
        return jsonify({"error": "Cannot modify your own admin status"}), 400
        
    user.is_admin = not user.is_admin
    db.session.commit()
    
    return jsonify({
        "message": f"User {user.email} {'promoted to admin' if user.is_admin else 'demoted from admin'} successfully",
        "is_admin": user.is_admin
    })

@admin_bp.route('/stats', methods=['GET'])
@admin_required
def get_stats():
    """Get system statistics (admin only)."""
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    admin_users = User.query.filter_by(is_admin=True).count()
    
    # You can add more stats as needed from other models
    
    return jsonify({
        "stats": {
            "total_users": total_users,
            "active_users": active_users,
            "admin_users": admin_users
        }
    })