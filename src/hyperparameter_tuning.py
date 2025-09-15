"""
Hyperparameter tuning module for Sign Language Digits Recognition
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import json
import os
from datetime import datetime
import time

from config import *
from model_training import ModelTrainer
from evaluation import ModelEvaluator


class HyperparameterTuner:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.results = {}

    def grid_search(self, param_grid, model_type='advanced', max_experiments=None):
        """
        Perform grid search over hyperparameters
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        if max_experiments and len(combinations) > max_experiments:
            # Randomly sample combinations if too many
            np.random.shuffle(combinations)
            combinations = combinations[:max_experiments]

        print(f"Starting grid search with {len(combinations)} combinations...")

        results = []

        for i, combination in enumerate(combinations):
            print(f"\nExperiment {i + 1}/{len(combinations)}")

            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            print(f"Parameters: {params}")

            try:
                # Train model with current parameters
                start_time = time.time()
                history = self.trainer.train_model(model_type=model_type, **params)
                training_time = time.time() - start_time

                # Evaluate model
                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                # Store results
                result = {
                    'experiment_id': i + 1,
                    'parameters': params,
                    'training_time': training_time,
                    'final_train_accuracy': history.history['accuracy'][-1],
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'test_accuracy': metrics['accuracy'],
                    'test_precision': metrics['precision_macro'],
                    'test_recall': metrics['recall_macro'],
                    'test_f1': metrics['f1_macro'],
                    'epochs_trained': len(history.history['accuracy'])
                }

                results.append(result)

                # Print current results
                print(f"Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"Training Time: {training_time:.2f} seconds")

            except Exception as e:
                print(f"Error in experiment {i + 1}: {str(e)}")
                continue

        self.results = results
        return results

    def random_search(self, param_distributions, n_iterations=20, model_type='advanced'):
        """
        Perform random search over hyperparameters
        """
        print(f"Starting random search with {n_iterations} iterations...")

        results = []

        for i in range(n_iterations):
            print(f"\nExperiment {i + 1}/{n_iterations}")

            # Sample random parameters
            params = {}
            for param_name, param_range in param_distributions.items():
                if isinstance(param_range, list):
                    params[param_name] = np.random.choice(param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        params[param_name] = np.random.uniform(param_range[0], param_range[1])

            print(f"Parameters: {params}")

            try:
                # Train model with current parameters
                start_time = time.time()
                history = self.trainer.train_model(model_type=model_type, **params)
                training_time = time.time() - start_time

                # Evaluate model
                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                # Store results
                result = {
                    'experiment_id': i + 1,
                    'parameters': params,
                    'training_time': training_time,
                    'final_train_accuracy': history.history['accuracy'][-1],
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'test_accuracy': metrics['accuracy'],
                    'test_precision': metrics['precision_macro'],
                    'test_recall': metrics['recall_macro'],
                    'test_f1': metrics['f1_macro'],
                    'epochs_trained': len(history.history['accuracy'])
                }

                results.append(result)

                print(f"Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"Training Time: {training_time:.2f} seconds")

            except Exception as e:
                print(f"Error in experiment {i + 1}: {str(e)}")
                continue

        self.results = results
        return results

    def analyze_batch_size_effect(self, batch_sizes=None, model_type='advanced'):
        """
        Analyze the effect of different batch sizes
        """
        if batch_sizes is None:
            batch_sizes = BATCH_SIZES

        results = []

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            params = DEFAULT_CONFIG.copy()
            params['batch_size'] = batch_size

            try:
                start_time = time.time()
                history = self.trainer.train_model(model_type=model_type, **params)
                training_time = time.time() - start_time

                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                results.append({
                    'batch_size': batch_size,
                    'test_accuracy': metrics['accuracy'],
                    'test_f1': metrics['f1_macro'],
                    'training_time': training_time,
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'epochs_trained': len(history.history['accuracy'])
                })

            except Exception as e:
                print(f"Error with batch size {batch_size}: {str(e)}")
                continue

        return results

    def analyze_learning_rate_effect(self, learning_rates=None, model_type='advanced'):
        """
        Analyze the effect of different learning rates
        """
        if learning_rates is None:
            learning_rates = LEARNING_RATES

        results = []

        for lr in learning_rates:
            print(f"\nTesting learning rate: {lr}")

            params = DEFAULT_CONFIG.copy()
            params['learning_rate'] = lr

            try:
                start_time = time.time()
                history = self.trainer.train_model(model_type=model_type, **params)
                training_time = time.time() - start_time

                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                results.append({
                    'learning_rate': lr,
                    'test_accuracy': metrics['accuracy'],
                    'test_f1': metrics['f1_macro'],
                    'training_time': training_time,
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'epochs_trained': len(history.history['accuracy'])
                })

            except Exception as e:
                print(f"Error with learning rate {lr}: {str(e)}")
                continue

        return results

    def analyze_regularization_effect(self, dropout_rates=None, l1_lambdas=None,
                                      l2_lambdas=None, model_type='advanced'):
        """
        Analyze the effect of different regularization parameters
        """
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES
        if l1_lambdas is None:
            l1_lambdas = L1_LAMBDAS
        if l2_lambdas is None:
            l2_lambdas = L2_LAMBDAS

        results = []

        # Test dropout rates
        print("Testing dropout rates...")
        for dropout_rate in dropout_rates:
            print(f"\nTesting dropout rate: {dropout_rate}")

            params = DEFAULT_CONFIG.copy()
            params['dropout_rate'] = dropout_rate

            try:
                history = self.trainer.train_model(model_type=model_type, **params)
                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                results.append({
                    'regularization_type': 'dropout',
                    'parameter_value': dropout_rate,
                    'test_accuracy': metrics['accuracy'],
                    'test_f1': metrics['f1_macro'],
                    'final_val_accuracy': history.history['val_accuracy'][-1]
                })

            except Exception as e:
                print(f"Error with dropout rate {dropout_rate}: {str(e)}")
                continue

        # Test L1 regularization
        print("\nTesting L1 regularization...")
        for l1_lambda in l1_lambdas:
            print(f"Testing L1 lambda: {l1_lambda}")

            params = DEFAULT_CONFIG.copy()
            params['l1_lambda'] = l1_lambda

            try:
                history = self.trainer.train_model(model_type=model_type, **params)
                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                results.append({
                    'regularization_type': 'l1',
                    'parameter_value': l1_lambda,
                    'test_accuracy': metrics['accuracy'],
                    'test_f1': metrics['f1_macro'],
                    'final_val_accuracy': history.history['val_accuracy'][-1]
                })

            except Exception as e:
                print(f"Error with L1 lambda {l1_lambda}: {str(e)}")
                continue

        # Test L2 regularization
        print("\nTesting L2 regularization...")
        for l2_lambda in l2_lambdas:
            print(f"Testing L2 lambda: {l2_lambda}")

            params = DEFAULT_CONFIG.copy()
            params['l2_lambda'] = l2_lambda

            try:
                history = self.trainer.train_model(model_type=model_type, **params)
                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                results.append({
                    'regularization_type': 'l2',
                    'parameter_value': l2_lambda,
                    'test_accuracy': metrics['accuracy'],
                    'test_f1': metrics['f1_macro'],
                    'final_val_accuracy': history.history['val_accuracy'][-1]
                })

            except Exception as e:
                print(f"Error with L2 lambda {l2_lambda}: {str(e)}")
                continue

        return results

    def plot_hyperparameter_analysis(self, results, param_name, save_path=None):
        """
        Plot hyperparameter analysis results
        """
        if not results:
            print("No results to plot")
            return

        df = pd.DataFrame(results)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Test accuracy vs parameter
        axes[0].plot(df[param_name], df['test_accuracy'], 'bo-')
        axes[0].set_xlabel(param_name)
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title(f'Test Accuracy vs {param_name}')
        axes[0].grid(True)

        # F1 score vs parameter
        axes[1].plot(df[param_name], df['test_f1'], 'ro-')
        axes[1].set_xlabel(param_name)
        axes[1].set_ylabel('Test F1-Score')
        axes[1].set_title(f'Test F1-Score vs {param_name}')
        axes[1].grid(True)

        # Training time vs parameter (if available)
        if 'training_time' in df.columns:
            axes[2].plot(df[param_name], df['training_time'], 'go-')
            axes[2].set_xlabel(param_name)
            axes[2].set_ylabel('Training Time (seconds)')
            axes[2].set_title(f'Training Time vs {param_name}')
            axes[2].grid(True)
        else:
            # Validation accuracy vs parameter
            axes[2].plot(df[param_name], df['final_val_accuracy'], 'mo-')
            axes[2].set_xlabel(param_name)
            axes[2].set_ylabel('Final Validation Accuracy')
            axes[2].set_title(f'Validation Accuracy vs {param_name}')
            axes[2].grid(True)

        if param_name in ['learning_rate', 'l1_lambda', 'l2_lambda']:
            for ax in axes:
                ax.set_xscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_regularization_comparison(self, results, save_path=None):
        """
        Plot comparison of different regularization techniques
        """
        if not results:
            print("No results to plot")
            return

        df = pd.DataFrame(results)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Test accuracy by regularization type
        sns.boxplot(data=df, x='regularization_type', y='test_accuracy', ax=axes[0])
        axes[0].set_title('Test Accuracy by Regularization Type')
        axes[0].set_ylabel('Test Accuracy')

        # Test F1 score by regularization type
        sns.boxplot(data=df, x='regularization_type', y='test_f1', ax=axes[1])
        axes[1].set_title('Test F1-Score by Regularization Type')
        axes[1].set_ylabel('Test F1-Score')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, results, filename=None):
        """
        Save hyperparameter tuning results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperparameter_results_{timestamp}.json"

        filepath = os.path.join(METRICS_DIR, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to: {filepath}")

        # Also save as CSV for easy analysis
        csv_filename = filename.replace('.json', '.csv')
        csv_filepath = os.path.join(METRICS_DIR, csv_filename)

        df = pd.DataFrame(results)
        df.to_csv(csv_filepath, index=False)
        print(f"Results also saved as CSV: {csv_filepath}")

    def find_best_parameters(self, results, metric='test_accuracy'):
        """
        Find the best parameters from tuning results
        """
        if not results:
            print("No results available")
            return None

        df = pd.DataFrame(results)
        best_idx = df[metric].idxmax()
        best_result = df.iloc[best_idx]

        print(f"\nBest parameters based on {metric}:")
        print(f"Score: {best_result[metric]:.4f}")

        if 'parameters' in best_result:
            print("Parameters:")
            for param, value in best_result['parameters'].items():
                print(f"  {param}: {value}")

        return best_result.to_dict()

    def compare_models(self, model_types=['basic', 'advanced', 'deep'], **kwargs):
        """
        Compare different model architectures
        """
        results = {}

        for model_type in model_types:
            print(f"\nTraining {model_type} model...")

            try:
                start_time = time.time()
                history = self.trainer.train_model(model_type=model_type, **kwargs)
                training_time = time.time() - start_time

                eval_results = self.trainer.evaluate_model()
                metrics = self.evaluator.calculate_metrics(
                    eval_results['y_true_classes'],
                    eval_results['y_pred_classes'],
                    eval_results['y_pred']
                )

                results[model_type] = {
                    'metrics': metrics,
                    'training_time': training_time,
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'epochs_trained': len(history.history['accuracy'])
                }

                print(f"Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"Training Time: {training_time:.2f} seconds")

            except Exception as e:
                print(f"Error training {model_type} model: {str(e)}")
                continue

        return results