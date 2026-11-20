"""
Retraining Pipeline for AI vs Human Media Detector
Triggered when feedback threshold is reached
Uses corrected labels to fine-tune the model
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config2 import MODELS_DIR, BACKEND_DIR
from hybrid_train2 import HybridTrainer


class Retrainer:
    """Handles retraining of the model using feedback data"""

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.model_path = self.models_dir / "image_classifier.keras"
        self.config_path = self.models_dir / "model_config.json"
        self.vocab_path = self.models_dir / "label_vocabulary.json"

    def get_feedback_data(self):
        """Get feedback data from database with corrected labels"""
        try:
            import sqlite3
            db_path = BACKEND_DIR / "analysis.db"

            conn = sqlite3.connect(str(db_path))
            query = """
                SELECT ma.filename, ma.file_path, f.corrected_label
                FROM feedback f
                JOIN media_analysis ma ON f.analysis_id = ma.id
                WHERE f.corrected_label IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) == 0:
                return None

            print(f"📊 Found {len(df)} feedback samples for retraining")
            return df

        except Exception as e:
            print(f"❌ Error getting feedback data: {e}")
            return None

    def prepare_retraining_dataset(self, feedback_df):
        """Prepare dataset for retraining from feedback"""
        # Convert to format expected by training script
        retrain_data = []

        for _, row in feedback_df.iterrows():
            label = 'fake' if row['corrected_label'] else 'real'
            retrain_data.append((label, row['file_path']))

        if not retrain_data:
            return None

        # Create dataframe
        df = pd.DataFrame(data=retrain_data, columns=['tag', 'img_name'])
        return df

    def fine_tune_model(self, retrain_df, original_model_path):
        """Fine-tune the existing model with new data"""
        print("\n🔄 Starting fine-tuning process...")

        # Load original model
        model = keras.models.load_model(str(original_model_path))
        print("✅ Loaded original model")

        # Load configuration
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        # Prepare data for fine-tuning
        trainer = HybridTrainer()
        label_processor = keras.layers.StringLookup(
            num_oov_indices=0,
            vocabulary=np.unique(retrain_df["tag"])
        )

        # Extract features from retraining data
        print("📊 Extracting features from feedback images...")
        features, labels = trainer.prepare_features(
            retrain_df, label_processor, augment=True
        )

        # Fine-tune the model (only last few layers)
        print("🔧 Fine-tuning model...")
        for layer in model.layers[:-3]:
            layer.trainable = False

        # Recompile with lower learning rate
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            metrics=["accuracy"]
        )

        # Train on feedback data
        history = model.fit(
            features, labels,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        return model, history

    def retrain(self):
        """Main retraining function"""
        print("=" * 60)
        print("🔄 STARTING RETRAINING PIPELINE")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Check if original model exists
        if not self.model_path.exists():
            print("❌ No original model found. Run hybrid_train.py first.")
            return False

        # Get feedback data
        feedback_df = self.get_feedback_data()
        if feedback_df is None or len(feedback_df) < 10:
            print("⚠️ Insufficient feedback data for retraining (need at least 10 samples)")
            return False

        # Prepare retraining dataset
        retrain_df = self.prepare_retraining_dataset(feedback_df)
        if retrain_df is None:
            print("❌ Failed to prepare retraining dataset")
            return False

        # Fine-tune model
        try:
            model, history = self.fine_tune_model(retrain_df, self.model_path)

            # Save backup of original model
            backup_path = self.models_dir / f"image_classifier_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            import shutil
            shutil.copy2(self.model_path, backup_path)
            print(f"💾 Original model backed up to {backup_path}")

            # Save fine-tuned model
            model.save(str(self.model_path))
            print(" Model retrained and saved successfully!")

            # Update config with retraining info
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            config['last_retrained'] = datetime.now().isoformat()
            config['retraining_samples'] = len(retrain_df)
            config['retraining_history'] = {
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_loss': float(history.history['loss'][-1])
            }

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print("=" * 60)
            print("✅ RETRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    retrainer = Retrainer()
    retrainer.retrain()