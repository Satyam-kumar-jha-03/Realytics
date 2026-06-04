"""
Authentication utilities for AI Detector
"""
import hashlib
import secrets
from functools import wraps
from flask import request, jsonify, session


def hash_password(password):
    """Hash a password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256()
    hash_obj.update((password + salt).encode('utf-8'))
    return f"{salt}:{hash_obj.hexdigest()}"


def verify_password(password, hashed):
    """Verify a password against its hash"""
    try:
        salt, hash_value = hashed.split(':')
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode('utf-8'))
        return hash_obj.hexdigest() == hash_value
    except:
        return False


def login_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Authentication required', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    return decorated_function