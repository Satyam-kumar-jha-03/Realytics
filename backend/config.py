
"""
Configuration file for AI vs Human Media Detector.
Uses relative paths and environment variables for all settings.
"""

import os
import sys
import logging
from pathlib import Path

# Automatically load environment variables from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    # Checks for .env in current directory or project root
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Helper functions for environment parsing
def get_bool_env(var_name: str, default: bool = False) -> bool:
    val = os.environ.get(var_name)
    if val is None:
        return default
    return val.strip().lower() in {'true', '1', 't', 'yes', 'y'}

def get_int_env(var_name: str, default: int) -> int:
    try:
        return int(os.environ.get(var_name, default))
    except (ValueError, TypeError):
        return default

def get_float_env(var_name: str, default: float) -> float:
    try:
        return float(os.environ.get(var_name, default))
    except (ValueError, TypeError):
        return default


# BASE PATHS (Resolved to absolute paths)
CONFIG_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CONFIG_DIR
PROJECT_ROOT = BACKEND_DIR.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = PROJECT_ROOT / "models"
UPLOAD_FOLDER = BACKEND_DIR / "uploads"
DATASET_DIR = PROJECT_ROOT / "dataset"
TEMP_DIR = BACKEND_DIR / "temp"

# Create required directories on launch
for directory in (UPLOAD_FOLDER, MODELS_DIR, DATASET_DIR, TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# APPLICATION & SERVER SETTINGS
ENV = os.environ.get('FLASK_ENV', os.environ.get('ENV', 'development')).lower()
DEBUG = get_bool_env('DEBUG', default=(ENV == 'development'))
DEMO_MODE = get_bool_env('DEMO_MODE', default=False)
PORT = get_int_env('PORT', 5000)
HOST = os.environ.get('HOST', '0.0.0.0')

# Allowed CORS origins
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')


# DATABASE CONFIGURATION
DEFAULT_SQLITE_PATH = BACKEND_DIR / "analysis.db"
DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}')

# Fix PostgreSQL URI scheme if using newer SQLAlchemy with older Heroku/Render configs
if DATABASE_URI.startswith("postgres://"):
    DATABASE_URI = DATABASE_URI.replace("postgres://", "postgresql://", 1)

SQLALCHEMY_TRACK_MODIFICATIONS = False


# AUTHENTICATION & SECURITY
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    if ENV == 'production':
        raise ValueError("CRITICAL: JWT_SECRET_KEY environment variable must be set in production!")
    JWT_SECRET_KEY = 'dev-insecure-secret-key-change-in-production'

JWT_EXPIRATION_HOURS = get_int_env('JWT_EXPIRATION_HOURS', 24)
FREE_TRIAL_LIMIT = get_int_env('FREE_TRIAL_LIMIT', 1)



# FILE UPLOAD & VALIDATION LIMITS
MAX_CONTENT_LENGTH = get_int_env('MAX_CONTENT_LENGTH', 50 * 1024 * 1024)  # Default 50MB

ALLOWED_IMAGE_MIMES = {
    'image/jpeg',
    'image/png',
    'image/webp'
}
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

ALLOWED_VIDEO_MIMES = {
    'video/mp4',
    'video/quicktime',
    'video/x-msvideo',
    'video/avi',
    'video/mpeg',
    'application/mp4',
    'application/octet-stream'
}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


# VIRUS SCANNING (ClamAV)
CLAMAV_ENABLED = get_bool_env('CLAMAV_ENABLED', default=False)
CLAMAV_SOCKET = os.environ.get('CLAMAV_SOCKET', '/var/run/clamav/clamd.ctl')
CLAMAV_HOST = os.environ.get('CLAMAV_HOST', 'localhost')
CLAMAV_PORT = get_int_env('CLAMAV_PORT', 3310)


# ML / INFERENCE HARDWARE & MODELS

# Device configuration: 'cuda', 'mps', or 'cpu'
DEVICE = os.environ.get('DEVICE', 'cpu')

MODEL_PATHS = {
    'classifier': MODELS_DIR / "image_classifier.keras",
    'config': MODELS_DIR / "model_config.json",
    'vocab': MODELS_DIR / "label_vocabulary.json"
}

# Ensemble weights (ResNet50, EfficientNet, ViT)
ENSEMBLE_WEIGHTS = {
    'resnet50': get_float_env('WEIGHT_RESNET', 0.4),
    'efficientnet': get_float_env('WEIGHT_EFFICIENTNET', 0.4),
    'vit': get_float_env('WEIGHT_VIT', 0.2)
}

# Retraining and feedback triggers
RETRAIN_THRESHOLD = get_int_env('RETRAIN_THRESHOLD', 50)


# TRAINING & PREPROCESSING CONFIGURATION
IMG_SIZE = get_int_env('IMG_SIZE', 224)
BATCH_SIZE = get_int_env('BATCH_SIZE', 128)
EPOCHS = get_int_env('EPOCHS', 50)
MAX_SEQ_LENGTH = get_int_env('MAX_SEQ_LENGTH', 1)
NUM_FEATURES = get_int_env('NUM_FEATURES', 2048)

SMALL_DATASET_THRESHOLD = 5000
MEDIUM_DATASET_THRESHOLD = 20000
VIDEO_MODE_THRESHOLD = 100


# VIDEO ANALYSIS SETTINGS
MAX_VIDEO_DURATION_SECONDS = get_int_env('MAX_VIDEO_DURATION_SECONDS', 300)
FRAME_SAMPLE_INTERVAL = get_int_env('FRAME_SAMPLE_INTERVAL', 10)
MAX_FRAMES_TO_ANALYZE = get_int_env('MAX_FRAMES_TO_ANALYZE', 200)



# GENERATIVE AI / EXPLANATION CONFIGURATION (Gemini)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
ENABLE_EXPLANATIONS = get_bool_env('ENABLE_EXPLANATIONS', default=bool(GEMINI_API_KEY))



# DIAGNOSTICS & LOGGING
def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("📋 SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Environment:            {ENV}")
    print(f"Project Root:           {PROJECT_ROOT}")
    print(f"Models Directory:       {MODELS_DIR}")
    print(f"Upload Folder:          {UPLOAD_FOLDER}")
    print(f"Database URI:           {DATABASE_URI}")
    print(f"Inference Device:       {DEVICE}")
    print(f"Demo Mode:              {DEMO_MODE}")
    print(f"ClamAV Enabled:         {CLAMAV_ENABLED}")
    print(f"Retrain Threshold:      {RETRAIN_THRESHOLD}")
    print(f"Ensemble Weights:       {ENSEMBLE_WEIGHTS}")
    print(f"Gemini Explanations:    {ENABLE_EXPLANATIONS} (Model: {GEMINI_MODEL})")
    print(f"Max Video Duration:     {MAX_VIDEO_DURATION_SECONDS}s")
    print(f"Max Frames to Analyze:  {MAX_FRAMES_TO_ANALYZE}")
    print("=" * 60)

