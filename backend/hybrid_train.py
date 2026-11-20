import os
import gc
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

# Hardware & Precision Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Enable mixed precision (fp16) for modern GPUs (Nvidia Turing/Ampere or newer)
        keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"✅ GPU available: {len(gpus)} | Mixed Precision: mixed_float16 enabled")
    except RuntimeError as e:
        print(f"GPU config error: {e}")

# Hyperparameters
IMG_SIZE = 256            # 256 preserves higher frequency details better than 224
BATCH_SIZE = 32
STAGE1_EPOCHS = 5        # Train only classification head
STAGE2_EPOCHS = 25       # Fine-tune backbone + head end-to-end
RANDOM_SEED = 42


class DataPipeline:
    """High-performance tf.data input pipeline with dynamic data augmentation."""

    @staticmethod
    def augment_image(image):
        # 1. Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # 2. Random contrast & brightness (mimicking varied generative lighting)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # 3. Random JPEG quality degradation (crucial for detecting compression resilience)
        image = tf.image.random_jpeg_quality(image, min_jpeg_quality=75, max_jpeg_quality=100)

        return tf.clip_by_value(image, 0.0, 255.0)

    @classmethod
    def load_and_preprocess(cls, file_path, label, augment=False):
        img_raw = tf.io.read_file(file_path)
        # Decode as 3-channel RGB image
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method='bicubic')
        img = tf.cast(img, tf.float32)

        if augment:
            img = cls.augment_image(img)

        # Preprocess using ConvNeXt / EfficientNet standard ([0, 255] or [-1, 1])
        img = keras.applications.convnext.preprocess_input(img)
        return img, tf.cast(label, tf.float32)

    @classmethod
    def create_dataset(cls, df, augment=False, shuffle=True):
        paths = df['img_name'].values
        labels = df['label'].values

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df), seed=RANDOM_SEED)

        dataset = dataset.map(
            lambda x, y: cls.load_and_preprocess(x, y, augment=augment),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset


class AIDetectorTrainer:
    """Full End-to-End Classifier with Two-Stage Transfer Learning."""

    def __init__(self):
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.label_mapping = {}

    def build_model(self):
        inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

        # Modern architecture: ConvNeXtBase or ConvNeXtSmall
        base_model = keras.applications.ConvNeXtSmall(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling=None
        )

        # Initially freeze backbone for Stage 1 head warm-up
        base_model.trainable = False

        # Head architecture preserving multi-scale spatial pooling
        x = base_model.output
        gap = keras.layers.GlobalAveragePooling2D()(x)
        gmp = keras.layers.GlobalMaxPooling2D()(x)
        # Concatenate average and max pooling to capture both background context and sharp artifacts
        x = keras.layers.Concatenate()([gap, gmp])

        x = keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("gelu")(x)
        x = keras.layers.Dropout(0.3)(x)

        x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("gelu")(x)
        x = keras.layers.Dropout(0.2)(x)

        # Output with float32 cast to preserve numerical stability under mixed precision
        outputs = keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="classifier_output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="AI_vs_Real_ConvNeXt")
        return model, base_model

    def train(self):
        print("=" * 60)
        print("STARTING AI VS REAL END-TO-END DETECTOR TRAINING")
        print("=" * 60)

        # Step 1: Prepare data CSVs
        train_df, val_df = self.prepare_data()
        if train_df is None:
            return

        print(f"Dataset summary: {len(train_df)} train, {len(val_df)} validation")

        train_ds = DataPipeline.create_dataset(train_df, augment=True, shuffle=True)
        val_ds = DataPipeline.create_dataset(val_df, augment=False, shuffle=False)

        # Step 2: Build Model
        model, base_model = self.build_model()
        model.summary()

        # Checkpoints & Callbacks
        checkpoint_path = str(self.models_dir / "best_ai_detector.keras")
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # STAGE 1: Train Classification Head (Backbone Frozen)
        print("\n" + "=" * 50)
        print("STAGE 1: Training Classification Head (Backbone Frozen)")
        print("=" * 50)

        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", keras.metrics.AUC(name="auc")]
        )

        history_stage1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE1_EPOCHS,
            callbacks=[checkpoint],
            verbose=1
        )

        
        # STAGE 2: Fine-Tune Backbone End-to-End
        print("\n" + "=" * 50)
        print("STAGE 2: Unfreezing Top Blocks for End-to-End Fine-Tuning")
        print("=" * 50)

        base_model.trainable = True
        # Keep earliest layers frozen (basic edges/colors), unfreeze the top ~40 layers
        for layer in base_model.layers[:-40]:
            layer.trainable = False

        # Cosine decay schedule for gentle fine-tuning
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=3e-5,
            decay_steps=STAGE2_EPOCHS * len(train_ds),
            alpha=0.05
        )

        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", keras.metrics.AUC(name="auc")]
        )

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=STAGE2_EPOCHS,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )

        # Step 3: Evaluation
        print("\nFinal Evaluation on Validation Data:")
        eval_metrics = model.evaluate(val_ds, verbose=1)
        for name, val in zip(model.metrics_names, eval_metrics):
            print(f"  {name}: {val:.4f}")

        # Save config
        self.save_artifacts(eval_metrics)

    def prepare_data(self):
        """Discovers dataset folders and creates balanced train/val splits."""
        current_dir = Path(__file__).parent
        dataset_path = current_dir.parent / "dataset"
        train_path = dataset_path / "train"

        if not train_path.exists():
            print(f"Training path {train_path} does not exist!")
            return None, None

        classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
        if "real" not in classes or "fake" not in classes:
            print(f"Expected 'real' and 'fake' directories. Found: {classes}")
            return None, None

        # Standardize: real = 0, fake/ai = 1
        self.label_mapping = {"real": 0, "fake": 1}

        records = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for tag in ["real", "fake"]:
            class_dir = train_path / tag
            files = [f for f in class_dir.rglob('*') if f.suffix.lower() in valid_exts]
            print(f"  📸 Found {len(files)} {tag} images")
            for f in files:
                records.append((str(f), self.label_mapping[tag]))

        df = pd.DataFrame(records, columns=['img_name', 'label'])
        train_df, val_df = train_test_split(
            df, test_size=0.15, random_state=RANDOM_SEED, stratify=df['label']
        )
        return train_df, val_df

    def save_artifacts(self, eval_metrics):
        config = {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'label_mapping': self.label_mapping,
            'val_loss': float(eval_metrics[0]),
            'val_accuracy': float(eval_metrics[1]),
            'backbone': 'ConvNeXtSmall',
            'precision': 'mixed_float16'
        }
        with open(self.models_dir / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration and weights saved in {self.models_dir}")


def main():
    keras.backend.clear_session()
    gc.collect()
    trainer = AIDetectorTrainer()
    trainer.train()


if __name__ == "__main__":
    main()