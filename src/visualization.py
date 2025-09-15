"""
Visualization module for Sign Language Digits Recognition
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os
from config import *


class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')

    def plot_sample_images(self, X_data, y_data, num_samples=20, save_path=None):
        """
        Plot sample images from each digit class
        """
        fig, axes = plt.subplots(2, 10, figsize=(20, 8))
        fig.suptitle('Sample Sign Language Digit Images', fontsize=16, y=0.95)

        for digit in range(10):
            # Find indices of current digit
            if len(y_data.shape) > 1:  # One-hot encoded
                digit_indices = np.where(np.argmax(y_data, axis=1) == digit)[0]
            else:  # Regular labels
                digit_indices = np.where(y_data == digit)[0]

            if len(digit_indices) > 0:
                # First row - first sample
                sample_idx = digit_indices[0]
                img = X_data[sample_idx]
                axes[0, digit].imshow(img)
                axes[0, digit].set_title(f'Digit {digit}', fontsize=12)
                axes[0, digit].axis('off')

                # Second row - second sample if available
                if len(digit_indices) > 1:
                    sample_idx = digit_indices[1]
                    img = X_data[sample_idx]
                    axes[1, digit].imshow(img)
                    axes[1, digit].axis('off')
                else:
                    axes[1, digit].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_data_distribution(self, y_data, save_path=None):
        """
        Plot class distribution in the dataset
        """
        if len(y_data.shape) > 1:  # One-hot encoded
            labels = np.argmax(y_data, axis=1)
        else:
            labels = y_data

        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))

        plt.figure(figsize=(12, 8))

        # Create bar plot
        bars = plt.bar(distribution.keys(), distribution.values(),
                       color=plt.cm.tab10(np.linspace(0, 1, len(distribution))))

        plt.xlabel('Digit Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution in Sign Language Digits Dataset', fontsize=14)
        plt.xticks(range(10))

        # Add count labels on bars
        for i, (digit, count) in enumerate(distribution.items()):
            plt.text(digit, count + max(counts) * 0.01, str(count),
                     ha='center', va='bottom', fontsize=10)

        # Add statistics text
        total_samples = sum(counts)
        mean_samples = np.mean(counts)
        std_samples = np.std(counts)

        stats_text = f'Total Samples: {total_samples}\n'
        stats_text += f'Mean per class: {mean_samples:.1f}\n'
        stats_text += f'Std per class: {std_samples:.1f}'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return distribution

    def plot_image_preprocessing_steps(self, original_img, processed_steps, save_path=None):
        """
        Visualize image preprocessing steps
        """
        n_steps = len(processed_steps) + 1
        fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))

        if n_steps == 1:
            axes = [axes]

        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Preprocessing steps
        for i, (step_name, step_img) in enumerate(processed_steps.items(), 1):
            if step_img.ndim == 2:  # Grayscale
                axes[i].imshow(step_img, cmap='gray')
            else:
                axes[i].imshow(step_img)
            axes[i].set_title(step_name)
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_augmentation_examples(self, datagen, X_sample, n_examples=8, save_path=None):
        """
        Show data augmentation examples
        """
        fig, axes = plt.subplots(2, n_examples, figsize=(2 * n_examples, 6))

        # Original images in first row
        for i in range(n_examples):
            axes[0, i].imshow(X_sample[i])
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')

        # Augmented images in second row
        aug_iter = datagen.flow(X_sample[:n_examples], batch_size=n_examples)
        aug_batch = next(aug_iter)

        for i in range(n_examples):
            axes[1, i].imshow(aug_batch[i])
            axes[1, i].set_title('Augmented')
            axes[1, i].axis('off')

        plt.suptitle('Data Augmentation Examples', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_architecture_comparison(self, model_summaries, save_path=None):
        """
        Compare model architectures
        """
        fig, axes = plt.subplots(len(model_summaries), 1, figsize=(12, 4 * len(model_summaries)))

        if len(model_summaries) == 1:
            axes = [axes]

        for i, (model_name, summary_info) in enumerate(model_summaries.items()):
            # Extract layer information
            layer_names = summary_info['layers']
            layer_params = summary_info['parameters']

            # Create horizontal bar chart
            y_pos = np.arange(len(layer_names))
            axes[i].barh(y_pos, layer_params)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(layer_names)
            axes[i].set_xlabel('Number of Parameters')
            axes[i].set_title(f'{model_name} Architecture')

            # Add parameter count as text
            for j, params in enumerate(layer_params):
                axes[i].text(params, j, f' {params:,}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_learning_curves_comparison(self, histories, save_path=None):
        """
        Compare learning curves from multiple training runs
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        metrics = ['accuracy', 'loss', 'precision', 'recall']
        titles = ['Accuracy', 'Loss', 'Precision', 'Recall']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]

            for name, history in histories.items():
                if metric in history:
                    epochs = range(1, len(history[metric]) + 1)
                    ax.plot(epochs, history[metric], label=f'{name} (train)', alpha=0.7)

                    val_metric = f'val_{metric}'
                    if val_metric in history:
                        ax.plot(epochs, history[val_metric],
                                label=f'{name} (val)', linestyle='--', alpha=0.7)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_hyperparameter_heatmap(self, results_df, param1, param2, metric='test_accuracy', save_path=None):
        """
        Create heatmap for hyperparameter combinations
        """
        # Pivot the results for heatmap
        heatmap_data = results_df.pivot_table(
            values=metric,
            index=param1,
            columns=param2,
            aggfunc='mean'
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                    cbar_kws={'label': metric})
        plt.title(f'{metric} Heatmap: {param1} vs {param2}')
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_maps(self, model, X_sample, layer_names=None, save_path=None):
        """
        Visualize feature maps from convolutional layers
        """
        if layer_names is None:
            # Get first few conv layers
            layer_names = [layer.name for layer in model.layers
                           if 'conv' in layer.name.lower()][:4]

        # Create model to output feature maps
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

        # Get activations
        activations = activation_model.predict(np.expand_dims(X_sample, axis=0))

        # Plot feature maps
        fig, axes = plt.subplots(len(layer_names), 8, figsize=(16, 4 * len(layer_names)))

        for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
            n_features = min(8, activation.shape[-1])

            for j in range(n_features):
                if len(layer_names) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                ax.imshow(activation[0, :, :, j], cmap='viridis')
                ax.set_title(f'{layer_name}\nFeature {j}')
                ax.axis('off')

        plt.suptitle('Convolutional Feature Maps', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_gradient_flow(self, model, X_sample, y_sample, save_path=None):
        """
        Visualize gradient flow through the network
        """
        import tensorflow as tf

        with tf.GradientTape() as tape:
            tape.watch(X_sample)
            predictions = model(np.expand_dims(X_sample, axis=0))
            loss = tf.keras.losses.categorical_crossentropy(
                np.expand_dims(y_sample, axis=0), predictions
            )

        # Get gradients
        gradients = tape.gradient(loss, X_sample)

        # Plot original image and gradients
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(X_sample)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Gradients
        grad_img = np.mean(np.abs(gradients), axis=-1)
        axes[1].imshow(grad_img, cmap='hot')
        axes[1].set_title('Gradient Magnitude')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(X_sample, alpha=0.7)
        axes[2].imshow(grad_img, cmap='hot', alpha=0.3)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_results_dashboard(self, results_dict, save_path=None):
        """
        Create a comprehensive results dashboard
        """
        fig = plt.figure(figsize=(20, 16))

        # Create subplots with different sizes
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Model comparison (top left)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['test_accuracy'] for model in models]

        bars = ax1.bar(models, accuracies, color=plt.cm.tab10(np.linspace(0, 1, len(models))))
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')

        # Training time comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2:4])
        training_times = [results_dict[model]['training_time'] for model in models]

        ax2.barh(models, training_times, color='lightblue')
        ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)')

        # Parameter count comparison (middle right)
        ax3 = fig.add_subplot(gs[1, 2:4])
        param_counts = [results_dict[model]['total_params'] for model in models]

        ax3.barh(models, param_counts, color='lightgreen')
        ax3.set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Parameters')

        # Precision-Recall comparison (bottom left)
        ax4 = fig.add_subplot(gs[2, 0:2])
        precisions = [results_dict[model]['precision'] for model in models]
        recalls = [results_dict[model]['recall'] for model in models]

        x = np.arange(len(models))
        width = 0.35

        ax4.bar(x - width / 2, precisions, width, label='Precision', alpha=0.8)
        ax4.bar(x + width / 2, recalls, width, label='Recall', alpha=0.8)
        ax4.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()

        # F1-Score comparison (bottom middle)
        ax5 = fig.add_subplot(gs[2, 2])
        f1_scores = [results_dict[model]['f1_score'] for model in models]

        ax5.pie(f1_scores, labels=models, autopct='%1.3f', startangle=90)
        ax5.set_title('F1-Score Distribution', fontsize=14, fontweight='bold')

        # ROC-AUC comparison (bottom right)
        ax6 = fig.add_subplot(gs[2, 3])
        if 'roc_auc' in results_dict[models[0]]:
            roc_aucs = [results_dict[model]['roc_auc'] for model in models]
            ax6.plot(models, roc_aucs, 'o-', linewidth=2, markersize=8)
            ax6.set_title('ROC-AUC Scores', fontsize=14, fontweight='bold')
            ax6.set_ylabel('AUC Score')
            ax6.set_ylim(0.8, 1.0)
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

        # Summary statistics (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')

        # Create summary table
        summary_data = []
        for model in models:
            summary_data.append([
                model,
                f"{results_dict[model]['test_accuracy']:.4f}",
                f"{results_dict[model]['precision']:.4f}",
                f"{results_dict[model]['recall']:.4f}",
                f"{results_dict[model]['f1_score']:.4f}",
                f"{results_dict[model]['training_time']:.1f}s"
            ])

        table = ax7.table(cellText=summary_data,
                          colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(models) + 1):
            for j in range(6):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

        plt.suptitle('Sign Language Digits Recognition - Results Dashboard',
                     fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_all_plots(self, base_dir=None):
        """
        Save all generated plots to specified directory
        """
        if base_dir is None:
            base_dir = PLOTS_DIR

        print(f"All plots saved to: {base_dir}")
        return True