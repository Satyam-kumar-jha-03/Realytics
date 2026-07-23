# wsgi.py - Place in repository root
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app from backend
from backend.app import app

# This is what gunicorn will use
application = app
