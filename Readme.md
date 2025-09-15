# Sign Language Digits Recognition using ConvNets

This project implements a comprehensive deep learning system for recognizing sign language digits (0-9) using Convolutional Neural Networks. The system includes data preprocessing, multiple CNN architectures, hyperparameter tuning, and extensive evaluation metrics.

## Project Structure

```
sign_language_recognition/
├── data/
│   ├── raw/                    # Original dataset (place CSV files here)
│   ├── processed/              # Preprocessed data
│   └── splits/                 # Train/validation/test splits
├── models/
│   ├── saved_models/           # Trained model checkpoints
│   └── model_architecture.py   # CNN model definitions
├── src/
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── model_training.py       # Training pipeline
│   ├── evaluation.py           # Model evaluation metrics
│   ├── visualization.py        # Plotting and visualization
│   └── hyperparameter_tuning.py # Hyperparameter experiments
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook for analysis
├── results/
│   ├── plots/                  # Generated plots and figures
│   ├── metrics/                # Evaluation results
│   └── reports/                # Performance reports
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
└── config.py                   # Configuration settings
```

## Features

### CNN Architectures
- **Basic CNN**: Simple 3-layer convolutional network
- **Advanced CNN**: Multi-block CNN with batch normalization
- **Deep CNN**: VGG-inspired deep architecture
- **ResNet-like**: Residual connections for better gradient flow

### Hyperparameter Analysis
- Batch size optimization
- Learning rate scheduling
- Dropout regularization
- L1/L2 weight regularization
- Early stopping patience
- Data normalization techniques

### Evaluation Metrics
- Confusion Matrix
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Precision-Recall curves
- Classification report heatmaps
- Misclassification analysis
- Prediction confidence analysis

### Visualization
- Sample image displays
- Class distribution analysis
- Training history plots
- Hyperparameter effect plots
- Results dashboard
- Feature map visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sign_language_recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download from: https://www.kaggle.com/ardamavi/sign-language-digits-dataset
   - Place `sign_mnist_train.csv` and `sign_mnist_test.csv` in `data/raw/` directory

## Usage

### Quick Start

Run the comprehensive analysis:
```bash
python main.py --mode comprehensive
```

### Individual Components

1. **Data Exploration Only**:
```bash
python main.py --mode explore
```

2. **Train Single Model**:
```bash
python main.py --mode train --model advanced
```

3. **Hyperparameter Tuning**:
```bash
python main.py --mode tune --tuning individual --experiments 20
```

4. **Model Architecture Comparison**:
```bash
python main.py --mode compare
```

### Jupyter Notebook Analysis

Start Jupyter and open the analysis notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Configuration

Modify `config.py` to adjust:
- Image dimensions and channels
- Hyperparameter ranges
- Model architecture parameters
- File paths and directories
- Training configurations

## Model Architectures

### Basic CNN
- 3 Convolutional blocks
- Max pooling layers
- Dense layers with dropout
- ~100K parameters

### Advanced CNN
- 4 Convolutional blocks
- Batch normalization
- Global average pooling
- Dropout regularization
- ~500K parameters

### Deep CNN
- VGG-inspired architecture
- 4 blocks with multiple conv layers
- Heavy regularization
- ~2M parameters

### ResNet-like
- Residual connections
- Skip connections
- Identity mappings
- ~1M parameters

## Hyperparameter Tuning

The system supports three tuning approaches:

1. **Grid Search**: Exhaustive search over parameter grid
2. **Random Search**: Random sampling from parameter distributions
3. **Individual Analysis**: Systematic analysis of each parameter

### Analyzed Parameters
- Batch sizes: [16, 32, 64, 128]
- Learning rates: [0.0001, 0.001, 0.01]
- Dropout rates: [0.2, 0.3, 0.5]
- L1/L2 regularization: [0.001, 0.01, 0.1]
- Epochs: [50, 100, 150]
- Patience values: [5, 10, 15]

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Curve Analysis
- **ROC Curves**: True positive rate vs False positive rate
- **AUC-ROC**: Area under ROC curve
- **Precision-Recall Curves**: Precision vs Recall trade-off

### Visualization
- **Confusion Matrix**: Class-wise prediction accuracy
- **Classification Report Heatmap**: Per-class metrics
- **Misclassification Analysis**: Common error patterns
- **Confidence Analysis**: Prediction certainty distribution

## Results Structure

Results are automatically saved in organized directories:

```
results/
├── experiment_name/
│   ├── plots/
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   └── precision_recall_curves.png
│   ├── metrics/
│   │   ├── evaluation_report.txt
│   │   └── hyperparameter_results.json
│   └── models/
│       └── best_model.h5
```

## Expected Performance

Typical performance metrics:
- **Basic CNN**: ~85-90% accuracy
- **Advanced CNN**: ~90-95% accuracy  
- **Deep CNN**: ~92-96% accuracy
- **ResNet-like**: ~93-97% accuracy

## Troubleshooting

### Common Issues

1. **Dataset not found**:
   - Ensure CSV files are in `data/raw/` directory
   - Check file names match exactly

2. **Memory errors**:
   - Reduce batch size in config.py
   - Use smaller model architectures

3. **GPU not detected**:
   - Install tensorflow-gpu
   - Check CUDA compatibility

4. **Import errors**:
   - Verify all dependencies installed
   - Check Python version compatibility

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with proper testing
4. Submit pull request with detailed description

## License

This project is for academic purposes as part of a Generative AI assignment.

## Acknowledgments

- Dataset: Kaggle Sign Language Digits Dataset
- TensorFlow/Keras for deep learning framework
- Scikit-learn for evaluation metrics
- Matplotlib/Seaborn for visualizations

## Citation

If using this code for academic purposes, please cite:
```
@misc{sign_language_digits_cnn,
  title={Sign Language Digits Recognition using ConvNets},
  author={[Your Name]},
  year={2024},
  howpublished={GitHub Repository}
}
```