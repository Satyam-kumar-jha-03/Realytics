"""
Authentication utilities for AI vs Human Media Forensics (Realytics)
Hardened password hashing with Werkzeug scrypt, timing attack resistance,
JWT token issuing and decoding, and request context population.
"""
import os
import re
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import request, jsonify, session, g
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

from config2 import JWT_SECRET_KEY, JWT_EXPIRATION_HOURS


def hash_password(password: str) -> str:
    """Hash a password using Werkzeug's scrypt implementation with secure salt."""
    return generate_password_hash(password, method='scrypt')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash using constant-time comparison."""
    if not hashed or not password:
        return False
    try:
        return check_password_hash(hashed, password)
    except Exception:
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password complexity:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one number
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one numerical digit."
    return True, "Password meets strength requirements."


def generate_token(user_id: str, email: str, role: str = 'analyst') -> str:
    """Generate a signed JWT token for an authenticated user."""
    now = datetime.now(timezone.utc)
    payload = {
        'user_id': str(user_id),
        'email': email,
        'role': role,
        'exp': now + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': now
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token."""
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
    except (jwt.PyJWTError, Exception):
        return None


def login_required(f):
    """
    Route decorator requiring authentication.
    Extracts Bearer token from 'Authorization' header first, then falls back to session.
    Populates flask.g.user_id, g.user_email, and g.user_role.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = None
        user_email = None
        user_role = 'analyst'

        # 1. Check Authorization header: 'Bearer <token>'
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1].strip()
            payload = decode_token(token)
            if payload:
                user_id = payload.get('user_id')
                user_email = payload.get('email')
                user_role = payload.get('role', 'analyst')

        # 2. Check query parameter fallback (convenience for direct browser links or print views)
        if not user_id and 'token' in request.args:
            payload = decode_token(request.args.get('token'))
            if payload:
                user_id = payload.get('user_id')
                user_email = payload.get('email')
                user_role = payload.get('role', 'analyst')

        # 3. Fall back to Flask session
        if not user_id and 'user_id' in session:
            user_id = session['user_id']
            user_email = session.get('email', '')
            user_role = session.get('role', 'analyst')

        if not user_id:
            return jsonify({
                'success': False,
                'error': 'Authentication required. Please sign in to access this resource.',
                'redirect': '/login.html'
            }), 401

        # Populate request context
        g.user_id = str(user_id)
        g.user_email = user_email
        g.user_role = user_role
        return f(*args, **kwargs)

    return decorated_function


def optional_auth(f):
    """
    Populates g.user_id and g.user_email if a valid token is provided,
    but does not reject unauthenticated requests.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.user_id = None
        g.user_email = None
        g.user_role = 'guest'

        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1].strip()
            payload = decode_token(token)
            if payload:
                g.user_id = str(payload.get('user_id'))
                g.user_email = payload.get('email')
                g.user_role = payload.get('role', 'analyst')
        elif 'user_id' in session:
            g.user_id = str(session['user_id'])
            g.user_email = session.get('email')
            g.user_role = session.get('role', 'analyst')

        return f(*args, **kwargs)

    return decorated_function

