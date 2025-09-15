"""
Configuration file for Sign Language Digits Recognition
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Dataset parameters
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 10
CHANNELS = 3

# Training parameters
BATCH_SIZES = [16, 32, 64, 128]
LEARNING_RATES = [0.001, 0.01, 0.0001]
EPOCHS_LIST = [50, 100, 150]
DROPOUT_RATES = [0.2, 0.3, 0.5]
PATIENCE_VALUES = [5, 10, 15]
L1_LAMBDAS = [0.001, 0.01, 0.1]
L2_LAMBDAS = [0.001, 0.01, 0.1]

# Default hyperparameters for initial training
DEFAULT_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'dropout_rate': 0.3,
    'patience': 10,
    'l1_lambda': 0.001,
    'l2_lambda': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1
}

# Model architecture parameters
CONV_FILTERS = [32, 64, 128, 256]
KERNEL_SIZES = [(3, 3), (5, 5)]
POOL_SIZES = [(2, 2)]
DENSE_UNITS = [128, 256, 512]

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
                 RESULTS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)