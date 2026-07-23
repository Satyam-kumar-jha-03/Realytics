# server.py - For Windows production server
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

if __name__ == "__main__":
    from waitress import serve
    
    print("=" * 60)
    print("🚀 REALYTICS Server (Windows - Waitress)")
    print("=" * 60)
    print(f"📍 http://localhost:5000")
    print(f"📁 Working Directory: {os.getcwd()}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    serve(app, host="0.0.0.0", port=5000, threads=4)