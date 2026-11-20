"""
AI vs Human Media & Text Detector - Main Flask Application
Hardened version with in-memory video processing, concurrency protections,
and modern SQLAlchemy / Python 3.12+ compatibility.
"""
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import subprocess
import hashlib
import sys
import threading
import time
import uuid
import cv2
from flask import Flask, jsonify, request, send_from_directory, session, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import magic
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
clean_filename = None  # Placeholder for secure_filename function

from auth2 import (
    hash_password, verify_password, validate_password_strength,
    generate_token, decode_token, login_required, optional_auth
)

# Import configuration
from config2 import (
    PROJECT_ROOT, FRONTEND_DIR, MODELS_DIR, UPLOAD_FOLDER,
    DATABASE_URI, MAX_CONTENT_LENGTH,
    ALLOWED_IMAGE_MIMES, ALLOWED_IMAGE_EXTS,
    ALLOWED_VIDEO_MIMES, ALLOWED_VIDEO_EXTS,
    DEMO_MODE, CLAMAV_ENABLED, CLAMAV_SOCKET, CLAMAV_HOST, CLAMAV_PORT,
    RETRAIN_THRESHOLD,
    GEMINI_API_KEY, GEMINI_MODEL, ENABLE_EXPLANATIONS,
    MAX_VIDEO_DURATION_SECONDS, FRAME_SAMPLE_INTERVAL, MAX_FRAMES_TO_ANALYZE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Optional ClamAV import
CLAMAV_AVAILABLE = False
if CLAMAV_ENABLED:
    try:
        import pyclamd
        CLAMAV_AVAILABLE = True
    except ImportError:
        logger.warning("pyclamd not installed. Disabling ClamAV.")
        CLAMAV_ENABLED = False

# Optional Gemini explanations
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    if ENABLE_EXPLANATIONS and GEMINI_API_KEY and not GEMINI_API_KEY.startswith("YOUR_"):
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        logger.info("Gemini API configured for explanations.")
    else:
        logger.info("Gemini explanations disabled or unconfigured.")
except ImportError:
    logger.warning("google-generativeai not installed. Explanations disabled.")

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

db = SQLAlchemy(app)

# Global lock & flag to prevent duplicate retraining subprocesses
retraining_lock = threading.Lock()
is_retraining = False

# Rate limiting data structure & thread safety
rate_limit_lock = threading.Lock()
rate_limits = {}

def is_rate_limited(key: str, max_requests: int = 15, window_seconds: int = 60) -> bool:
    """Simple sliding-window rate limiter per key (IP, user ID, or endpoint)."""
    now = time.time()
    with rate_limit_lock:
        timestamps = rate_limits.get(key, [])
        timestamps = [t for t in timestamps if now - t < window_seconds]
        if len(timestamps) >= max_requests:
            rate_limits[key] = timestamps
            return True
        timestamps.append(now)
        rate_limits[key] = timestamps
        return False

# Security Headers & Request Limiters
@app.after_request
def apply_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File payload exceeds the allowed 50MB limit.'
    }), 413

# ============================================================================
# DATABASE MODELS
# ============================================================================

def utc_now():
    return datetime.now(timezone.utc)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(36), primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='analyst')
    created_at = db.Column(db.DateTime, default=utc_now)
    last_login = db.Column(db.DateTime, nullable=True)

    analyses = db.relationship('MediaAnalysis', backref='owner', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class MediaAnalysis(db.Model):
    __tablename__ = 'media_analysis'
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True, index=True)
    filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(10), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    is_ai = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text, nullable=True)
    file_hash = db.Column(db.String(64), nullable=True)
    file_size_bytes = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now)

    feedbacks = db.relationship('Feedback', backref='analysis', cascade='all, delete-orphan', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'media_type': self.media_type,
            'is_ai': self.is_ai,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'file_hash': self.file_hash,
            'file_size_bytes': self.file_size_bytes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.String(36), primary_key=True)
    analysis_id = db.Column(db.String(36), db.ForeignKey('media_analysis.id', ondelete='CASCADE'), nullable=False)
    feedback_type = db.Column(db.String(10), nullable=False)
    corrected_label = db.Column(db.Boolean, nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now)


# FILE VALIDATION
def validate_file_mime(file_stream, allowed_mimes):
    try:
        header = file_stream.read(2048)
        file_stream.seek(0)
        mime = magic.from_buffer(header, mime=True)
        return mime, mime in allowed_mimes
    except Exception as e:
        logger.error(f"MIME detection error: {e}")
        return None, False


def validate_file_clamav(file_stream):
    if not CLAMAV_ENABLED or not CLAMAV_AVAILABLE:
        return True, "ClamAV not enabled"

    cd = None
    try:
        cd = pyclamd.ClamdUnixSocket(CLAMAV_SOCKET)
    except Exception:
        try:
            cd = pyclamd.ClamdNetworkSocket(CLAMAV_HOST, CLAMAV_PORT)
        except Exception:
            logger.warning("Could not connect to ClamAV daemon. Skipping virus check.")
            return True, "ClamAV unavailable"

    try:
        file_stream.seek(0)
        result = cd.scan_stream(file_stream.read())
        file_stream.seek(0)
        if result:
            return False, f"Malware detected: {result}"
        return True, "Clean"
    except Exception as e:
        logger.error(f"ClamAV scan error: {e}")
        return True, "Scan error"


def validate_file(filename, file_stream):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return False, f"Invalid image extension: {ext}."

    mime, is_allowed = validate_file_mime(file_stream, ALLOWED_IMAGE_MIMES)
    if not is_allowed:
        return False, f"Invalid MIME type: {mime}."

    if CLAMAV_ENABLED:
        is_clean, message = validate_file_clamav(file_stream)
        if not is_clean:
            return False, message

    return True, "Valid"


def validate_video_file(filename, file_stream):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return False, f"Invalid video extension: {ext}."

    if CLAMAV_ENABLED:
        is_clean, message = validate_file_clamav(file_stream)
        if not is_clean:
            return False, message

    return True, "Valid"


# MODEL PREDICTOR
class AIModelPredictor:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.model_path = self.models_dir / "image_classifier.keras"
        self.config_path = self.models_dir / "model_config.json"
        self.vocab_path = self.models_dir / "label_vocabulary.json"

        self.model = None
        self.feature_extractor = None
        self.preprocess_fn = None
        self.class_names = []
        self.img_size = (224, 224)
        self.is_initialized = False
        self.use_fallback = False
        self.demo_mode = DEMO_MODE

        self.metrics = {'total_predictions': 0, 'total_time': 0.0, 'avg_time': 0.0}

        if not self.demo_mode:
            self._load_model()
        else:
            logger.info("DEMO MODE ACTIVE - Random predictions will be served.")
            self.is_initialized = True
            self.use_fallback = True
    def _load_model(self):
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file missing at {self.model_path}. Fallback active.")
                self.use_fallback = True
                self.is_initialized = True
                return

            self.model = keras.models.load_model(str(self.model_path))
            self.model.trainable = False

            # Derive img_size from the model itself — single source of truth
            if len(self.model.input_shape) == 4:
                _, h, w, _ = self.model.input_shape
                self.img_size = (h, w)
                logger.info(f"✅ Model loaded. input={self.model.input_shape}, "
                            f"output={self.model.output_shape}, img_size={self.img_size}")
            else:
                logger.error(f"❌ Model input_shape {self.model.input_shape} is not 4D. "
                            f"Fallback active.")
                self.use_fallback = True
                self.is_initialized = True
                return

            self.preprocess_fn = keras.applications.convnext.preprocess_input
            self.class_names = ["real", "fake"]
            self.is_initialized = True
            self.use_fallback = False
        except Exception as e:
            logger.exception(f"Model initialization failed: {e}")
            self.use_fallback = True
            self.is_initialized = True

    def _preprocess_numpy_image(self, img_array):
        """Converts raw RGB array directly to model input without disk I/O."""
        img = Image.fromarray(img_array).resize(self.img_size)
        arr = np.array(img, dtype=np.float32)
        preprocessed = self.preprocess_fn(arr.copy())
        return preprocessed[None, ...]

    def predict_image_array(self, img_array):
        if self.demo_mode:
            return self._demo_prediction()
        if self.use_fallback or self.model is None:
            return self._fallback_prediction()

        try:
            img = tf.convert_to_tensor(img_array, dtype=tf.uint8)
            img = tf.image.resize(img, self.img_size, method='bicubic')
            img = tf.cast(img, tf.float32)
            img = keras.applications.convnext.preprocess_input(img)
            batch_input = tf.expand_dims(img, axis=0)

            out = self.model(batch_input, training=False).numpy()

            # Handle both 1-output sigmoid and 2-output softmax
            if out.shape[-1] == 1:
                prob_fake = float(out[0][0])
            else:
                prob_fake = float(out[0][1])

            is_ai = prob_fake > 0.5
            confidence = float(max(prob_fake, 1.0 - prob_fake) * 100)

            return {
                'isAI': is_ai,
                'confidence': confidence,
                'probabilities': {
                    'real': float((1.0 - prob_fake) * 100),
                    'fake': float(prob_fake * 100),
                },
                'predicted_class': 'fake' if is_ai else 'real',
            }
        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            return self._fallback_prediction()

        
    def predict_image(self, image_path):
        start_time = time.time()
        self.metrics['total_predictions'] += 1

        if self.demo_mode:
            return self._demo_prediction()
        if self.use_fallback or self.model is None:
            return self._fallback_prediction()

        try:
            with Image.open(image_path) as img:
                rgb_img = np.array(img.convert("RGB"))
            
            result = self.predict_image_array(rgb_img)
            elapsed = time.time() - start_time
            self.metrics['total_time'] += elapsed
            self.metrics['avg_time'] = self.metrics['total_time'] / self.metrics['total_predictions']
            result['prediction_time'] = elapsed
            return result
        except Exception as e:
            logger.error(f"Prediction error for {image_path}: {e}")
            return self._fallback_prediction()

    def predict_video(self, video_path, frame_interval=FRAME_SAMPLE_INTERVAL, max_frames=MAX_FRAMES_TO_ANALYZE):
        if self.demo_mode:
            return self._demo_video_prediction()
        if self.use_fallback or self.model is None:
            return self._fallback_video_prediction()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._fallback_video_prediction()

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frames_to_sample = total_frames // max(1, frame_interval)
            if frames_to_sample > max_frames:
                frame_interval = max(1, total_frames // max_frames)

            frame_predictions = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # In-memory inference without saving files to disk
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = self.predict_image_array(frame_rgb)
                    if res:
                        frame_predictions.append({
                            'frame': frame_count,
                            'isAI': res['isAI'],
                            'confidence': res['confidence']
                        })
                frame_count += 1

            if not frame_predictions:
                return self._fallback_video_prediction()

            # Maintain odd count for breaking ties
            if len(frame_predictions) % 2 == 0 and len(frame_predictions) > 1:
                frame_predictions.pop(random.randint(0, len(frame_predictions) - 1))

            ai_count = sum(1 for p in frame_predictions if p['isAI'])
            human_count = len(frame_predictions) - ai_count
            overall_is_ai = ai_count > human_count

            winning_confs = [p['confidence'] for p in frame_predictions if p['isAI'] == overall_is_ai]
            overall_confidence = sum(winning_confs) / len(winning_confs) if winning_confs else 50.0

            return {
                'isAI': overall_is_ai,
                'confidence': overall_confidence,
                'frames_analyzed': len(frame_predictions),
                'ai_frames': ai_count,
                'human_frames': human_count,
                'frame_results': frame_predictions,
                'sampling_interval': frame_interval,
                'video_duration_sec': duration
            }
        finally:
            cap.release()

    def _demo_prediction(self):
        is_ai = random.random() > 0.5
        conf = round(random.uniform(70.0, 98.0), 2)
        return {
            'isAI': is_ai,
            'confidence': conf,
            'probabilities': {'real': 100 - conf, 'fake': conf},
            'predicted_class': 'fake' if is_ai else 'real'
        }

    def _demo_video_prediction(self):
        is_ai = random.random() > 0.5
        conf = round(random.uniform(70.0, 98.0), 2)
        frames = random.randint(10, 30)
        ai_f = random.randint(0, frames)
        return {
            'isAI': is_ai,
            'confidence': conf,
            'frames_analyzed': frames,
            'ai_frames': ai_f,
            'human_frames': frames - ai_f
        }

    def _fallback_prediction(self):
        return {
            'isAI': False,
            'confidence': 50.0,
            'probabilities': {'Real': 50.0, 'Fake': 50.0},
            'predicted_class': 'Unknown'
        }

    def _fallback_video_prediction(self):
        return {
            'isAI': False,
            'confidence': 50.0,
            'frames_analyzed': 0,
            'ai_frames': 0,
            'human_frames': 0
        }

# Instantiate model
model_predictor = AIModelPredictor()

# ANALYSIS ROUTINES

def analyze_text(text_content):
    if DEMO_MODE:
        is_ai = random.random() > 0.5
        return is_ai, round(random.uniform(70.0, 95.0), 2)

    text_content = text_content.strip()
    if len(text_content) < 50:
        return False, 50.0

    sentences = [s.strip() for s in re.split(r'[.!?]+', text_content) if s.strip()]
    words = text_content.lower().split()
    if not words:
        return False, 50.0

    lexical_diversity = len(set(words)) / max(len(words), 1)

    ai_indicators = 0
    human_indicators = 0

    sentence_lengths = [len(sent.split()) for sent in sentences]
    if sentence_lengths:
        variance = np.var(sentence_lengths)
        if variance < 8:
            ai_indicators += 1
        else:
            human_indicators += 1

    if lexical_diversity > 0.75:
        ai_indicators += 1
    elif lexical_diversity < 0.45:
        human_indicators += 1

    formal_words = {'utilization', 'methodology', 'optimization', 'implementation', 'furthermore'}
    informal_words = {'awesome', 'cool', 'gonna', 'wanna', 'kinda', 'btw'}
    formal_count = sum(1 for w in words if w in formal_words)
    informal_count = sum(1 for w in words if w in informal_words)

    if formal_count > informal_count:
        ai_indicators += 1
    else:
        human_indicators += 1

    total = ai_indicators + human_indicators
    ai_score = ai_indicators / total if total > 0 else 0.5
    is_ai = ai_score > 0.5
    confidence = min(95.0, max(55.0, float(ai_score * 100)))

    return is_ai, confidence


def generate_explanation(media_type, is_ai, confidence, text_content=None):
    if not ENABLE_EXPLANATIONS or not GEMINI_AVAILABLE:
        return "Automated explanation is disabled or not configured."

    classification = "AI-GENERATED" if is_ai else "REAL/HUMAN-CREATED"
    if media_type == 'text':
        preview = (text_content or "")[:250].replace('\n', ' ')
        prompt = (
            f"An algorithmic detector classified this text as {classification} with {confidence:.1f}% confidence.\n"
            f"Text snippet: \"{preview}\"\n"
            "Provide a concise 2-sentence explanation highlighting tone, syntax uniformity, or markers."
        )
    else:
        prompt = (
            f"An image/video was classified as {classification} with {confidence:.1f}% confidence.\n"
            "Provide a concise 2-sentence explanation of visual cues (lighting, edge consistency, texture patterns) "
            "typical for this decision."
        )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "Explanation unavailable at this time."

# ============================================================================
# ROUTES & AUTHENTICATION ENDPOINTS
# ============================================================================

@app.route('/')
def serve_frontend():
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(str(FRONTEND_DIR), path)


# --- Authentication Routes ---

@app.route('/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json() or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''

        # Email regex format check
        if not email or not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            return jsonify({'success': False, 'error': 'A valid email address is required.'}), 400

        # Password complexity validation
        valid_pwd, msg = validate_password_strength(password)
        if not valid_pwd:
            return jsonify({'success': False, 'error': msg}), 400

        # Check existing user
        if db.session.query(User).filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'An account with this email already exists.'}), 409

        # Create user
        new_user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(password),
            role='analyst',
            created_at=utc_now(),
            last_login=utc_now()
        )
        db.session.add(new_user)
        db.session.commit()

        token = generate_token(new_user.id, new_user.email, new_user.role)
        return jsonify({
            'success': True,
            'token': token,
            'user_id': new_user.id,
            'email': new_user.email,
            'role': new_user.role
        }), 201

    except Exception as e:
        logger.exception("Registration error")
        return jsonify({'success': False, 'error': 'Failed to complete registration.'}), 500


@app.route('/auth/login', methods=['POST'])
def login():
    try:
        client_ip = request.remote_addr or 'unknown'
        if is_rate_limited(f"login_{client_ip}", max_requests=10, window_seconds=60):
            return jsonify({
                'success': False,
                'error': 'Too many login attempts. Please wait 60 seconds before retrying.'
            }), 429

        data = request.get_json() or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''

        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password are required.'}), 400

        user = db.session.query(User).filter_by(email=email).first()
        if not user or not verify_password(password, user.password_hash):
            return jsonify({'success': False, 'error': 'Invalid credentials.'}), 401

        user.last_login = utc_now()
        db.session.commit()

        token = generate_token(user.id, user.email, user.role)
        return jsonify({
            'success': True,
            'token': token,
            'user_id': user.id,
            'email': user.email,
            'role': user.role
        })

    except Exception as e:
        logger.exception("Login error")
        return jsonify({'success': False, 'error': 'Authentication server error.'}), 500


@app.route('/auth/demo-login', methods=['POST'])
def demo_login():
    """Quick-start guest analyst session for reviewers without registration."""
    try:
        demo_email = "analyst@realytics.io"
        user = db.session.query(User).filter_by(email=demo_email).first()
        if not user:
            user = User(
                id="demo-analyst-uuid-001",
                email=demo_email,
                password_hash=hash_password("DemoPassword123!"),
                role='lead_analyst',
                created_at=utc_now(),
                last_login=utc_now()
            )
            db.session.add(user)
            db.session.commit()
        else:
            user.last_login = utc_now()
            db.session.commit()

        token = generate_token(user.id, user.email, user.role)
        return jsonify({
            'success': True,
            'token': token,
            'user_id': user.id,
            'email': user.email,
            'role': user.role,
            'is_demo': True
        })
    except Exception as e:
        logger.exception("Demo login error")
        return jsonify({'success': False, 'error': 'Unable to initialize demo session.'}), 500


@app.route('/auth/me', methods=['GET'])
@login_required
def get_current_user():
    try:
        user = db.session.get(User, g.user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found.'}), 404
        return jsonify({'success': True, 'user': user.to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Session cleared.'})


# --- Forensics & Inference Endpoints ---

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_media():
    try:
        # Rate-limiting check per user (max 30 requests per minute)
        if is_rate_limited(f"analyze_{g.user_id}", max_requests=30, window_seconds=60):
            return jsonify({
                'success': False,
                'error': 'Inference rate limit exceeded. Please wait a minute before submitting further scans.'
            }), 429

        media_type = request.form.get('type', '').lower()
        file_hash = None
        file_size_bytes = 0

        if media_type == 'text':
            text_content = request.form.get('text', '').strip()
            if not text_content:
                return jsonify({'success': False, 'error': 'No text provided'}), 400

            file_id = str(uuid.uuid4())
            safe_name = f"text_{file_id[:8]}.txt"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)

            encoded_text = text_content.encode('utf-8')
            file_hash = hashlib.sha256(encoded_text).hexdigest()
            file_size_bytes = len(encoded_text)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            is_ai, confidence = analyze_text(text_content)
            explanation = generate_explanation('text', is_ai, confidence, text_content=text_content)
            filename = safe_name

        elif 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify({'success': False, 'error': 'Empty or missing file'}), 400

            clean_filename = secure_filename(file.filename) or f"upload_{uuid.uuid4().hex[:8]}"
            ext = os.path.splitext(clean_filename)[1].lower()
            if not media_type:
                media_type = 'video' if ext in ALLOWED_VIDEO_EXTS else 'image'

            if media_type == 'image':
                valid, msg = validate_file(clean_filename, file.stream)
            else:
                valid, msg = validate_video_file(clean_filename, file.stream)

            if not valid:
                return jsonify({'success': False, 'error': msg}), 400

            file_id = str(uuid.uuid4())
            filename = f"{file_id}{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Read stream into memory to calculate hash & save
            file.stream.seek(0)
            file_bytes = file.stream.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            file_size_bytes = len(file_bytes)

            with open(file_path, 'wb') as f:
                f.write(file_bytes)

            if media_type == 'video':
                cap = cv2.VideoCapture(file_path)
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = total / fps if fps > 0 else 0
                finally:
                    cap.release()

                if duration > MAX_VIDEO_DURATION_SECONDS:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({
                        'success': False,
                        'error': f'Video exceeds max limit of {MAX_VIDEO_DURATION_SECONDS}s.'
                    }), 400

                res = model_predictor.predict_video(file_path)
            else:
                res = model_predictor.predict_image(file_path)

            is_ai = res['isAI']
            confidence = res['confidence']
            explanation = generate_explanation(media_type, is_ai, confidence)

        else:
            return jsonify({'success': False, 'error': 'Missing file or text payload'}), 400

        record = MediaAnalysis(
            id=file_id,
            user_id=g.user_id,
            filename=clean_filename if 'file' in request.files else filename,
            media_type=media_type,
            file_path=file_path,
            is_ai=is_ai,
            confidence=confidence,
            explanation=explanation,
            file_hash=file_hash,
            file_size_bytes=file_size_bytes
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            'success': True,
            'isAI': is_ai,
            'confidence': float(confidence),
            'file_id': file_id,
            'file_hash': file_hash,
            'filename': record.filename
        })

    except Exception as e:
        logger.exception("Inference endpoint error")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/explain/<analysis_id>', methods=['GET'])
@login_required
def get_explanation(analysis_id):
    try:
        analysis = db.session.get(MediaAnalysis, analysis_id)
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404

        # Access check: user owns analysis, or analysis is demo/unowned, or user is admin
        if analysis.user_id and analysis.user_id != g.user_id and g.user_role != 'admin':
            return jsonify({'success': False, 'error': 'Unauthorized to view this analysis'}), 403

        if not analysis.explanation:
            text_snippet = None
            if analysis.media_type == 'text' and os.path.exists(analysis.file_path):
                with open(analysis.file_path, 'r', encoding='utf-8') as f:
                    text_snippet = f.read()

            analysis.explanation = generate_explanation(
                analysis.media_type, analysis.is_ai, analysis.confidence, text_content=text_snippet
            )
            db.session.commit()

        return jsonify({
            'success': True,
            'id': analysis.id,
            'filename': analysis.filename,
            'media_type': analysis.media_type,
            'explanation': analysis.explanation,
            'is_ai': analysis.is_ai,
            'confidence': analysis.confidence,
            'file_hash': analysis.file_hash,
            'file_size_bytes': analysis.file_size_bytes,
            'created_at': analysis.created_at.isoformat() if analysis.created_at else None
        })
    except Exception as e:
        logger.exception("Explanation lookup failure")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    global is_retraining
    try:
        data = request.get_json() or {}
        analysis_id = data.get('analysis_id')
        feedback_type = data.get('feedback_type')
        corrected_label = data.get('corrected_label')

        if not analysis_id or feedback_type not in ('up', 'down'):
            return jsonify({'success': False, 'error': 'Invalid feedback specification'}), 400

        analysis = db.session.get(MediaAnalysis, analysis_id)
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis record not found'}), 404

        fb = Feedback(
            id=str(uuid.uuid4()),
            analysis_id=analysis_id,
            feedback_type=feedback_type,
            corrected_label=corrected_label
        )
        db.session.add(fb)
        db.session.commit()

        if corrected_label is not None:
            corrected_count = db.session.query(Feedback).filter(Feedback.corrected_label.isnot(None)).count()
            if corrected_count >= RETRAIN_THRESHOLD:
                with retraining_lock:
                    if not is_retraining:
                        retrain_script = Path(__file__).parent / "retrain.py"
                        if retrain_script.exists():
                            subprocess.Popen([sys.executable, str(retrain_script)])
                            is_retraining = True
                            logger.info("Retraining subprocess dispatched.")

        return jsonify({'success': True, 'message': 'Feedback registered'})
    except Exception as e:
        logger.exception("Feedback submission error")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history', methods=['GET'])
@login_required
def get_history():
    try:
        # Fetch records belonging to the current user
        records = (
            db.session.query(MediaAnalysis)
            .filter(
                (MediaAnalysis.user_id == g.user_id) |
                (MediaAnalysis.user_id.is_(None) & (g.user_id == 'demo-analyst-uuid-001'))
            )
            .order_by(MediaAnalysis.created_at.desc())
            .limit(100)
            .all()
        )
        return jsonify([r.to_dict() for r in records])
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analysis/<analysis_id>', methods=['DELETE'])
@login_required
def delete_analysis(analysis_id):
    try:
        analysis = db.session.get(MediaAnalysis, analysis_id)
        if not analysis:
            return jsonify({'success': False, 'error': 'Analysis not found'}), 404

        if analysis.user_id != g.user_id and g.user_role != 'admin':
            return jsonify({'success': False, 'error': 'Unauthorized to delete this record'}), 403

        # Remove local file if exists
        if analysis.file_path and os.path.exists(analysis.file_path):
            try:
                os.remove(analysis.file_path)
            except Exception as fe:
                logger.warning(f"Could not remove file: {fe}")

        # Delete associated feedback records first
        db.session.query(Feedback).filter_by(analysis_id=analysis_id).delete()

        db.session.delete(analysis)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Analysis deleted successfully.'})

    except Exception as e:
        logger.exception("Deletion failure")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    status = "healthy" if (model_predictor.is_initialized and not model_predictor.use_fallback) else "degraded"
    return jsonify({
        'status': status,
        'demo_mode': DEMO_MODE,
        'model_initialized': model_predictor.is_initialized,
        'fallback_active': model_predictor.use_fallback,
        'predictions_count': model_predictor.metrics['total_predictions']
    })


# APPLICATION ENTRY POINT & DEMO SEEDING
with app.app_context():
    db.create_all()
    # Seed default demo account if not existing
    demo_user = db.session.query(User).filter_by(email="analyst@realytics.io").first()
    if not demo_user:
        demo_user = User(
            id="demo-analyst-uuid-001",
            email="analyst@realytics.io",
            password_hash=hash_password("DemoPassword123!"),
            role="lead_analyst",
            created_at=utc_now(),
            last_login=utc_now()
        )
        db.session.add(demo_user)
        db.session.commit()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)