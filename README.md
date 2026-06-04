REALYTICS - AI Content Detection Platform
A comprehensive Flask-based backend for detecting AI-generated content (images, videos, and text) with user authentication, trial mode, feedback collection, model retraining, and confidence calibration analytics.

Features: 
A) Core Detection Capabilities
  1) Image Analysis - Detect AI-generated images using ResNet50-based classifier
  2) Video Analysis - Frame-sampling detection for AI-generated videos
  3) Text Analysis - Basic pattern detection (extensible with Gemini API)
   
B) User System
  1) User Registration & Login - Secure password hashing (SHA-256 with salt)
  2) Session-based Authentication - Simple session management without Flask-Login
  3) Anonymous Trial Mode - Session-tracked free trials for non-registered users
  4) User Dashboard - History of analyses with detailed results

C) Advanced Features
  1) AI Explanations - Gemini API integration (falls back to simple explanations)
  2) PDF Report Generation - Download detailed analysis reports
  3) Feedback Collection - User feedback for model improvement
  4) Retraining Pipeline - Fine-tune model using corrected feedback
  5) Confidence Calibration - Reliability diagrams, ECE, Brier score metrics
  6) Model Health Monitoring - Status endpoint for frontend integration

D) Frontend Pages
  1) Marketing Homepage - Demo upload, trust badges, FAQ
  2) Login Page - Responsive split-screen authentication
  3) Registration Page - Account creation with validation
  4) Dashboard - History, filters, delete, expandable results
