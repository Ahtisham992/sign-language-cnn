"""
Model training module for Sign Language Digits Recognition
"""
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from models.model_architecture import CNNModels
from data_preprocessing import DataPreprocessor


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.preprocessor = DataPreprocessor()

    def load_data(self, use_processed=True):
        """
        Load training data
        """
        if use_processed and self.preprocessor.load_processed_data():
            print("Using processed data")
            return True
        else:
            print("Loading and processing raw data...")
            if self.preprocessor.load_kaggle_dataset():
                self.preprocessor.normalize_data('standard')
                self.preprocessor.split_data(
                    test_size=DEFAULT_CONFIG['test_split'],
                    val_size=DEFAULT_CONFIG['validation_split']
                )
                self.preprocessor.save_processed_data()
                return True
            else:
                print("Failed to load data")
                return False

    def create_model(self, model_type='advanced', **kwargs):
        """
        Create and compile model
        """
        hyperparams = DEFAULT_CONFIG.copy()
        hyperparams.update(kwargs)

        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)

        if model_type == 'basic':
            self.model = CNNModels.create_basic_cnn(
                input_shape=input_shape,
                num_classes=NUM_CLASSES,
                dropout_rate=hyperparams['dropout_rate'],
                l1_lambda=hyperparams['l1_lambda'],
                l2_lambda=hyperparams['l2_lambda']
            )
        elif model_type == 'advanced':
            self.model = CNNModels.create_advanced_cnn(
                input_shape=input_shape,
                num_classes=NUM_CLASSES,
                dropout_rate=hyperparams['dropout_rate'],
                l1_lambda=hyperparams['l1_lambda'],
                l2_lambda=hyperparams['l2_lambda']
            )
        elif model_type == 'deep':
            self.model = CNNModels.create_deep_cnn(
                input_shape=input_shape,
                num_classes=NUM_CLASSES,
                dropout_rate=hyperparams['dropout_rate'],
                l1_lambda=hyperparams['l1_lambda'],
                l2_lambda=hyperparams['l2_lambda']
            )
        elif model_type == 'resnet':
            self.model = CNNModels.create_resnet_like(
                input_shape=input_shape,
                num_classes=NUM_CLASSES,
                dropout_rate=hyperparams['dropout_rate'],
                l1_lambda=hyperparams['l1_lambda'],
                l2_lambda=hyperparams['l2_lambda']
            )

        # Compile model
        CNNModels.compile_model(
            self.model,
            learning_rate=hyperparams['learning_rate']
        )

        return self.model

    def setup_callbacks(self, model_name='model', patience=10):
        """
        Setup training callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.h5"
        model_path = os.path.join(MODELS_DIR, model_filename)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

    def train_model(self, model_type='advanced', use_augmentation=True, **kwargs):
        """
        Train the model with given hyperparameters
        """
        if not hasattr(self.preprocessor, 'X_train') or self.preprocessor.X_train is None:
            if not self.load_data():
                raise ValueError("Failed to load training data")

        # Create model
        hyperparams = DEFAULT_CONFIG.copy()
        hyperparams.update(kwargs)

        self.create_model(model_type, **hyperparams)

        # Setup callbacks
        callbacks = self.setup_callbacks(
            model_name=f"{model_type}_cnn",
            patience=hyperparams['patience']
        )

        # Prepare training data
        if use_augmentation:
            datagen = self.preprocessor.create_data_augmentation()
            train_generator = datagen.flow(
                self.preprocessor.X_train,
                self.preprocessor.y_train,
                batch_size=hyperparams['batch_size']
            )

            # Calculate steps per epoch
            steps_per_epoch = len(self.preprocessor.X_train) // hyperparams['batch_size']

            # Train with data augmentation
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=hyperparams['epochs'],
                validation_data=(self.preprocessor.X_val, self.preprocessor.y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without data augmentation
            self.history = self.model.fit(
                self.preprocessor.X_train,
                self.preprocessor.y_train,
                batch_size=hyperparams['batch_size'],
                epochs=hyperparams['epochs'],
                validation_data=(self.preprocessor.X_val, self.preprocessor.y_val),
                callbacks=callbacks,
                verbose=1
            )

        return self.history

    def evaluate_model(self, test_data=None):
        """
        Evaluate model on test data
        """
        if test_data is None:
            X_test = self.preprocessor.X_test
            y_test = self.preprocessor.y_test
        else:
            X_test, y_test = test_data

        # Evaluate
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        return {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'y_pred': y_pred,
            'y_pred_classes': y_pred_classes,
            'y_true_classes': y_true_classes
        }

    def plot_training_history(self, save_path=None):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_training_history(self, filename=None):
        """
        Save training history to file
        """
        if self.history is None:
            print("No training history to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_history_{timestamp}.json"

        history_path = os.path.join(METRICS_DIR, filename)

        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"Training history saved to: {history_path}")

    def load_model(self, model_path):
        """
        Load a saved model
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        return self.model