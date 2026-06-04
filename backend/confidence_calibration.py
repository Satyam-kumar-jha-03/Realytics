"""
Confidence Calibration for AI Detector
Tracks prediction confidence vs actual accuracy
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io


class ConfidenceCalibrator:
    """Track and visualize model calibration"""
    
    def __init__(self, db, Feedback, MediaAnalysis, app):
        self.db = db
        self.Feedback = Feedback
        self.MediaAnalysis = MediaAnalysis
        self.app = app
        
    def collect_calibration_data(self, media_type=None, days=30):
        """Collect feedback data with actual vs predicted labels"""
        with self.app.app_context():
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = self.MediaAnalysis.query.filter(
                self.MediaAnalysis.created_at > cutoff_date
            )
            
            if media_type:
                query = query.filter(self.MediaAnalysis.media_type == media_type)
            
            analyses = query.all()
            
            data = []
            for analysis in analyses:
                feedbacks = self.Feedback.query.filter_by(analysis_id=analysis.id).all()
                
                if feedbacks and feedbacks[0].corrected_label is not None:
                    actual_is_ai = feedbacks[0].corrected_label
                    predicted_is_ai = analysis.is_ai
                    confidence = analysis.confidence
                    
                    data.append({
                        'analysis_id': analysis.id,
                        'media_type': analysis.media_type,
                        'predicted_is_ai': predicted_is_ai,
                        'actual_is_ai': actual_is_ai,
                        'confidence': confidence,
                        'is_correct': predicted_is_ai == actual_is_ai,
                        'created_at': analysis.created_at
                    })
            
            return pd.DataFrame(data)
    
    def compute_calibration_curve(self, df, n_bins=10):
        """Compute calibration curve (reliability diagram)"""
        if len(df) < 10:
            return None, None, None
        
        confidences = df['confidence'].values / 100.0
        correctness = df['is_correct'].astype(int).values
        
        prob_true, prob_pred = calibration_curve(correctness, confidences, n_bins=n_bins, strategy='uniform')
        
        return prob_true, prob_pred, len(df)
    
    def generate_calibration_plot(self, media_type=None):
        """Generate calibration plot for dashboard"""
        df = self.collect_calibration_data(media_type)
        
        if len(df) < 10:
            return None, "Insufficient feedback data (need at least 10 samples)"
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0a0a0a')
        
        prob_true, prob_pred, n_samples = self.compute_calibration_curve(df)
        
        if prob_true is not None:
            axes[0].plot(prob_pred, prob_true, 'o-', label='Model', color='#d4af37', linewidth=2, markersize=8)
            axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5, color='#888')
            axes[0].set_xlabel('Mean Predicted Confidence', color='#e0e0e0')
            axes[0].set_ylabel('Fraction of Positives (Accuracy)', color='#e0e0e0')
            axes[0].set_title(f'Reliability Diagram ({n_samples} samples)', color='#d4af37')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_facecolor('#1a1a1a')
        
        confidences = df['confidence'].values
        bins = np.linspace(0, 100, 11)
        axes[1].hist(confidences, bins=bins, color='#d4af37', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Confidence (%)', color='#e0e0e0')
        axes[1].set_ylabel('Frequency', color='#e0e0e0')
        axes[1].set_title('Confidence Distribution', color='#d4af37')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_facecolor('#1a1a1a')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer, "Success"
    
    def get_calibration_metrics(self, media_type=None):
        """Get calibration error metrics"""
        df = self.collect_calibration_data(media_type)
        
        if len(df) < 10:
            return {
                'has_data': False,
                'message': 'Insufficient data for calibration metrics',
                'total_samples': len(df)
            }
        
        prob_true, prob_pred, _ = self.compute_calibration_curve(df)
        if prob_true is not None:
            ece = np.mean(np.abs(prob_true - prob_pred)) * 100
        else:
            ece = None
        
        if prob_true is not None:
            mce = np.max(np.abs(prob_true - prob_pred)) * 100
        else:
            mce = None
        
        confidences = df['confidence'].values / 100.0
        correctness = df['is_correct'].astype(int).values
        brier_score = np.mean((confidences - correctness) ** 2)
        
        accuracy = df['is_correct'].mean() * 100
        avg_confidence = df['confidence'].mean()
        calibration_gap = avg_confidence - accuracy
        
        return {
            'has_data': True,
            'total_samples': len(df),
            'accuracy': round(accuracy, 1),
            'avg_confidence': round(avg_confidence, 1),
            'calibration_gap': round(calibration_gap, 1),
            'ece': round(ece, 1) if ece else None,
            'mce': round(mce, 1) if mce else None,
            'brier_score': round(brier_score, 3),
            'media_type': media_type or 'all'
        }
    
    def get_confidence_trend(self, days=30):
        """Track calibration over time"""
        with self.app.app_context():
            cutoff_date = datetime.now() - timedelta(days=days)
            
            analyses = self.MediaAnalysis.query.filter(
                self.MediaAnalysis.created_at > cutoff_date
            ).all()
            
            daily_data = {}
            for analysis in analyses:
                date_key = analysis.created_at.strftime('%Y-%m-%d')
                feedbacks = self.Feedback.query.filter_by(analysis_id=analysis.id).all()
                
                if feedbacks and feedbacks[0].corrected_label is not None:
                    if date_key not in daily_data:
                        daily_data[date_key] = {'correct': 0, 'total': 0, 'confidence_sum': 0}
                    
                    is_correct = analysis.is_ai == feedbacks[0].corrected_label
                    daily_data[date_key]['correct'] += int(is_correct)
                    daily_data[date_key]['total'] += 1
                    daily_data[date_key]['confidence_sum'] += analysis.confidence
            
            dates = sorted(daily_data.keys())
            accuracies = []
            confidences = []
            
            for date in dates:
                data = daily_data[date]
                accuracies.append(data['correct'] / data['total'] * 100)
                confidences.append(data['confidence_sum'] / data['total'])
            
            return dates, accuracies, confidences