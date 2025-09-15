"""
Main execution script for Sign Language Digits Recognition
"""
import os
import sys
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import *
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.hyperparameter_tuning import HyperparameterTuner
from src.visualization import Visualizer


def setup_experiment(experiment_name):
    """
    Setup experiment directory and logging
    """
    experiment_dir = os.path.join(RESULTS_DIR, experiment_name)
    plots_dir = os.path.join(experiment_dir, 'plots')
    metrics_dir = os.path.join(experiment_dir, 'metrics')
    models_dir = os.path.join(experiment_dir, 'models')

    for directory in [experiment_dir, plots_dir, metrics_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)

    return experiment_dir, plots_dir, metrics_dir, models_dir


def run_data_exploration():
    """
    Run data exploration and visualization
    """
    print("=" * 60)
    print("DATA EXPLORATION AND PREPROCESSING")
    print("=" * 60)

    # Initialize preprocessor and visualizer
    preprocessor = DataPreprocessor()
    visualizer = Visualizer()

    # Load data
    print("\n1. Loading dataset...")
    if not preprocessor.load_kaggle_dataset():
        print("Failed to load dataset. Please check the data directory.")
        return False

    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(preprocessor.image_data)}")
    print(f"Image shape: {preprocessor.image_data.shape}")
    print(f"Labels shape: {preprocessor.labels.shape}")

    # Visualize sample images
    print("\n2. Visualizing sample images...")
    visualizer.plot_sample_images(
        preprocessor.image_data,
        preprocessor.labels,
        save_path=os.path.join(PLOTS_DIR, 'sample_images.png')
    )

    # Show class distribution
    print("\n3. Analyzing class distribution...")
    distribution = visualizer.plot_data_distribution(
        preprocessor.labels,
        save_path=os.path.join(PLOTS_DIR, 'class_distribution.png')
    )

    print("Class distribution:")
    for digit, count in distribution.items():
        print(f"  Digit {digit}: {count} samples")

    # Normalize and split data
    print("\n4. Preprocessing data...")
    preprocessor.normalize_data('standard')
    preprocessor.split_data()

    # Save processed data
    preprocessor.save_processed_data()

    print("Data preprocessing completed!")
    return True


def train_single_model(model_type='advanced', experiment_name='single_model'):
    """
    Train a single model with default parameters
    """
    print("=" * 60)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("=" * 60)

    # Setup experiment directories
    exp_dir, plots_dir, metrics_dir, models_dir = setup_experiment(experiment_name)

    # Initialize trainer and evaluator
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    visualizer = Visualizer()

    # Load data
    print("\n1. Loading preprocessed data...")
    if not trainer.load_data():
        print("No preprocessed data found. Running data exploration first...")
        if not run_data_exploration():
            return False
        trainer.load_data()

    # Train model
    print("\n2. Training model...")
    start_time = time.time()
    history = trainer.train_model(model_type=model_type, use_augmentation=True)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")

    # Plot training history
    print("\n3. Plotting training history...")
    trainer.plot_training_history(save_path=os.path.join(plots_dir, 'training_history.png'))
    trainer.save_training_history()

    # Evaluate model
    print("\n4. Evaluating model...")
    eval_results = trainer.evaluate_model()

    # Calculate comprehensive metrics
    metrics = evaluator.calculate_metrics(
        eval_results['y_true_classes'],
        eval_results['y_pred_classes'],
        eval_results['y_pred']
    )

    # Generate evaluation plots
    print("\n5. Generating evaluation plots...")

    # Confusion matrix
    evaluator.plot_confusion_matrix(
        eval_results['y_true_classes'],
        eval_results['y_pred_classes'],
        save_path=os.path.join(plots_dir, 'confusion_matrix.png')
    )

    # ROC curves
    evaluator.plot_roc_curves(
        eval_results['y_true_classes'],
        eval_results['y_pred'],
        save_path=os.path.join(plots_dir, 'roc_curves.png')
    )

    # Precision-Recall curves
    evaluator.plot_precision_recall_curves(
        eval_results['y_true_classes'],
        eval_results['y_pred'],
        save_path=os.path.join(plots_dir, 'precision_recall_curves.png')
    )

    # Classification report heatmap
    evaluator.plot_classification_report_heatmap(
        metrics['classification_report'],
        save_path=os.path.join(plots_dir, 'classification_report_heatmap.png')
    )

    # Prediction confidence analysis
    evaluator.plot_prediction_confidence(
        eval_results['y_pred'],
        eval_results['y_true_classes'],
        save_path=os.path.join(plots_dir, 'prediction_confidence.png')
    )

    # Misclassification analysis
    evaluator.analyze_misclassifications(
        trainer.preprocessor.X_test,
        eval_results['y_true_classes'],
        eval_results['y_pred_classes'],
        eval_results['y_pred'],
        save_path=os.path.join(plots_dir, 'misclassifications.png')
    )

    # Generate comprehensive report
    print("\n6. Generating evaluation report...")
    evaluator.generate_evaluation_report(
        model_name=f"{model_type}_cnn",
        y_true=eval_results['y_true_classes'],
        y_pred=eval_results['y_pred_classes'],
        y_pred_proba=eval_results['y_pred'],
        training_time=training_time,
        save_path=os.path.join(metrics_dir, f'{model_type}_evaluation_report.txt')
    )

    print(f"\nModel training and evaluation completed!")
    print(f"Results saved to: {exp_dir}")

    # Return results for further analysis
    return {
        'model': trainer.model,
        'history': history,
        'metrics': metrics,
        'training_time': training_time,
        'eval_results': eval_results
    }


def run_hyperparameter_tuning(tuning_type='grid', max_experiments=20):
    """
    Run hyperparameter tuning experiments
    """
    print("=" * 60)
    print(f"HYPERPARAMETER TUNING - {tuning_type.upper()} SEARCH")
    print("=" * 60)

    # Setup experiment directories
    exp_dir, plots_dir, metrics_dir, models_dir = setup_experiment(f'hyperparameter_tuning_{tuning_type}')

    # Initialize tuner
    tuner = HyperparameterTuner()
    visualizer = Visualizer()

    # Load data
    if not tuner.trainer.load_data():
        print("No preprocessed data found. Running data exploration first...")
        if not run_data_exploration():
            return False

    results = []

    if tuning_type == 'grid':
        print("\n1. Running grid search...")
        param_grid = {
            'batch_size': [16, 32, 64],
            'learning_rate': [0.0001, 0.001, 0.01],
            'dropout_rate': [0.2, 0.3, 0.5]
        }
        results = tuner.grid_search(param_grid, max_experiments=max_experiments)

    elif tuning_type == 'random':
        print("\n1. Running random search...")
        param_distributions = {
            'batch_size': [16, 32, 64, 128],
            'learning_rate': (0.0001, 0.01),
            'dropout_rate': (0.1, 0.6),
            'l1_lambda': (0.0001, 0.1),
            'l2_lambda': (0.0001, 0.1)
        }
        results = tuner.random_search(param_distributions, n_iterations=max_experiments)

    elif tuning_type == 'individual':
        print("\n1. Running individual parameter analysis...")

        # Batch size analysis
        print("\n1a. Analyzing batch size effect...")
        batch_results = tuner.analyze_batch_size_effect()
        visualizer.plot_hyperparameter_analysis(
            batch_results,
            'batch_size',
            save_path=os.path.join(plots_dir, 'batch_size_analysis.png')
        )

        # Learning rate analysis
        print("\n1b. Analyzing learning rate effect...")
        lr_results = tuner.analyze_learning_rate_effect()
        visualizer.plot_hyperparameter_analysis(
            lr_results,
            'learning_rate',
            save_path=os.path.join(plots_dir, 'learning_rate_analysis.png')
        )

        # Regularization analysis
        print("\n1c. Analyzing regularization effect...")
        reg_results = tuner.analyze_regularization_effect()
        visualizer.plot_regularization_comparison(
            reg_results,
            save_path=os.path.join(plots_dir, 'regularization_analysis.png')
        )

        # Combine all results
        results = batch_results + lr_results + reg_results

    # Save results
    print("\n2. Saving tuning results...")
    tuner.save_results(results)

    # Find best parameters
    print("\n3. Finding best parameters...")
    best_params = tuner.find_best_parameters(results)

    print(f"\nHyperparameter tuning completed!")
    print(f"Results saved to: {exp_dir}")

    return results, best_params


def compare_model_architectures():
    """
    Compare different model architectures
    """
    print("=" * 60)
    print("MODEL ARCHITECTURE COMPARISON")
    print("=" * 60)

    # Setup experiment directories
    exp_dir, plots_dir, metrics_dir, models_dir = setup_experiment('model_comparison')

    # Initialize components
    tuner = HyperparameterTuner()
    evaluator = ModelEvaluator()
    visualizer = Visualizer()

    # Load data
    if not tuner.trainer.load_data():
        print("No preprocessed data found. Running data exploration first...")
        if not run_data_exploration():
            return False

    # Compare different architectures
    print("\n1. Training different model architectures...")
    model_types = ['basic', 'advanced', 'deep', 'resnet']

    results = tuner.compare_models(model_types, epochs=50)  # Reduced epochs for comparison

    # Generate comparison plots
    print("\n2. Generating comparison plots...")
    comparison_df = evaluator.compare_models(
        results,
        save_path=os.path.join(plots_dir, 'model_comparison.png')
    )

    # Create results dashboard
    print("\n3. Creating results dashboard...")

    # Prepare results for dashboard
    dashboard_results = {}
    for model_type, result in results.items():
        dashboard_results[model_type] = {
            'test_accuracy': result['metrics']['accuracy'],
            'precision': result['metrics']['precision_macro'],
            'recall': result['metrics']['recall_macro'],
            'f1_score': result['metrics']['f1_macro'],
            'training_time': result['training_time'],
            'total_params': 1000000,  # Placeholder - would get from model
        }

    visualizer.create_results_dashboard(
        dashboard_results,
        save_path=os.path.join(plots_dir, 'results_dashboard.png')
    )

    print(f"\nModel comparison completed!")
    print(f"Results saved to: {exp_dir}")

    return results, comparison_df


def run_comprehensive_analysis():
    """
    Run comprehensive analysis including all experiments
    """
    print("=" * 80)
    print("COMPREHENSIVE SIGN LANGUAGE DIGITS RECOGNITION ANALYSIS")
    print("=" * 80)

    all_results = {}

    # 1. Data exploration
    print("\nStep 1: Data exploration and preprocessing")
    if not run_data_exploration():
        return False

    # 2. Single model training
    print("\nStep 2: Training baseline model")
    baseline_results = train_single_model('advanced', 'baseline_advanced')
    all_results['baseline'] = baseline_results

    # 3. Hyperparameter tuning
    print("\nStep 3: Hyperparameter optimization")
    tuning_results, best_params = run_hyperparameter_tuning('individual', max_experiments=15)
    all_results['tuning'] = {'results': tuning_results, 'best_params': best_params}

    # 4. Model architecture comparison
    print("\nStep 4: Model architecture comparison")
    arch_results, comparison_df = compare_model_architectures()
    all_results['architectures'] = {'results': arch_results, 'comparison': comparison_df}

    # 5. Final model training with best parameters
    print("\nStep 5: Training final optimized model")
    if best_params and 'parameters' in best_params:
        final_results = train_single_model('advanced', 'final_optimized')
        all_results['final'] = final_results

    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETED!")
    print("=" * 80)
    print("\nResults summary:")
    if 'baseline' in all_results:
        print(f"Baseline accuracy: {all_results['baseline']['metrics']['accuracy']:.4f}")
    if 'final' in all_results:
        print(f"Final accuracy: {all_results['final']['metrics']['accuracy']:.4f}")

    print(f"\nAll results saved to: {RESULTS_DIR}")

    return all_results


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description='Sign Language Digits Recognition')
    parser.add_argument('--mode', type=str, choices=['explore', 'train', 'tune', 'compare', 'comprehensive'],
                        default='comprehensive', help='Execution mode')
    parser.add_argument('--model', type=str, choices=['basic', 'advanced', 'deep', 'resnet'],
                        default='advanced', help='Model architecture')
    parser.add_argument('--tuning', type=str, choices=['grid', 'random', 'individual'],
                        default='individual', help='Tuning method')
    parser.add_argument('--experiments', type=int, default=20,
                        help='Maximum number of tuning experiments')

    args = parser.parse_args()

    # Create necessary directories
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
                      RESULTS_DIR, PLOTS_DIR, METRICS_DIR]:
        os.makedirs(directory, exist_ok=True)

    print("Sign Language Digits Recognition System")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")

    try:
        if args.mode == 'explore':
            run_data_exploration()

        elif args.mode == 'train':
            train_single_model(args.model)

        elif args.mode == 'tune':
            run_hyperparameter_tuning(args.tuning, args.experiments)

        elif args.mode == 'compare':
            compare_model_architectures()

        elif args.mode == 'comprehensive':
            run_comprehensive_analysis()

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nExecution completed.")


if __name__ == "__main__":
    main()