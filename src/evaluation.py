"""
Evaluation module for Sign Language Digits Recognition
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.preprocessing import label_binarize
import os
from config import *


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')

        # Per-class metrics
        metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None)
        metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None)
        metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None)

        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=[f'Digit_{i}' for i in range(NUM_CLASSES)],
            output_dict=True
        )

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=[f'Digit {i}' for i in range(NUM_CLASSES)],
                    yticklabels=[f'Digit {i}' for i in range(NUM_CLASSES)])
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return cm

    def plot_classification_report_heatmap(self, classification_report_dict, save_path=None):
        """
        Plot classification report as heatmap
        """
        # Extract data from classification report
        data = []
        labels = []

        for class_name, metrics in classification_report_dict.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if isinstance(metrics, dict):
                data.append([metrics['precision'], metrics['recall'], metrics['f1-score']])
                labels.append(class_name)

        data = np.array(data)

        plt.figure(figsize=(8, 10))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=['Precision', 'Recall', 'F1-Score'],
                    yticklabels=labels)
        plt.title('Classification Report Heatmap')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_roc_auc(self, y_true, y_pred_proba):
        """
        Calculate ROC-AUC for multi-class classification
        """
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculate micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Calculate macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(NUM_CLASSES):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= NUM_CLASSES
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return fpr, tpr, roc_auc

    def plot_roc_curves(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curves for multi-class classification
        """
        fpr, tpr, roc_auc = self.calculate_roc_auc(y_true, y_pred_proba)

        plt.figure(figsize=(12, 8))

        # Plot ROC curve for each class
        colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
        for i, color in enumerate(colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Digit {i} (AUC = {roc_auc[i]:.2f})')

        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        # Plot macro-average ROC curve
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=4)

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Sign Language Digit Recognition')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return roc_auc

    def plot_precision_recall_curves(self, y_true, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curves for multi-class classification
        """
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

        plt.figure(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
        for i, color in enumerate(colors):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            pr_auc = auc(recall, precision)

            plt.plot(recall, precision, color=color, lw=2,
                     label=f'Digit {i} (AUC = {pr_auc:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Sign Language Digit Recognition')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_prediction_confidence(self, y_pred_proba, y_true, save_path=None):
        """
        Plot prediction confidence distribution
        """
        # Get max confidence for each prediction
        max_confidence = np.max(y_pred_proba, axis=1)
        predicted_labels = np.argmax(y_pred_proba, axis=1)
        correct_predictions = (predicted_labels == y_true)

        plt.figure(figsize=(12, 6))

        # Confidence distribution for correct vs incorrect predictions
        plt.subplot(1, 2, 1)
        plt.hist(max_confidence[correct_predictions], bins=20, alpha=0.7,
                 label='Correct Predictions', color='green', density=True)
        plt.hist(max_confidence[~correct_predictions], bins=20, alpha=0.7,
                 label='Incorrect Predictions', color='red', density=True)
        plt.xlabel('Maximum Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Confidence by class
        plt.subplot(1, 2, 2)
        class_confidences = []
        class_labels = []

        for i in range(NUM_CLASSES):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_confidences.extend(max_confidence[class_mask])
                class_labels.extend([f'Digit {i}'] * np.sum(class_mask))

        df_confidence = pd.DataFrame({
            'Confidence': class_confidences,
            'Class': class_labels
        })

        sns.boxplot(data=df_confidence, x='Class', y='Confidence')
        plt.xticks(rotation=45)
        plt.title('Confidence by Class')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_misclassifications(self, X_test, y_true, y_pred, y_pred_proba, save_path=None):
        """
        Analyze misclassified samples
        """
        misclassified_indices = np.where(y_true != y_pred)[0]

        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return

        # Show some misclassified examples
        num_examples = min(20, len(misclassified_indices))
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()

        for i in range(num_examples):
            idx = misclassified_indices[i]
            image = X_test[idx]
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_proba[idx, pred_label]

            axes[i].imshow(image)
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}')
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_examples, 20):
            axes[i].axis('off')

        plt.suptitle('Misclassified Examples', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Create confusion pairs analysis
        confusion_pairs = {}
        for idx in misclassified_indices:
            pair = (y_true[idx], y_pred[idx])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

        print("\nMost Common Misclassification Pairs:")
        print("(True Label, Predicted Label): Count")
        for (true_label, pred_label), count in sorted_pairs[:10]:
            print(f"(Digit {true_label}, Digit {pred_label}): {count}")

        return sorted_pairs

    def generate_evaluation_report(self, model_name, y_true, y_pred, y_pred_proba=None,
                                   training_time=None, save_path=None):
        """
        Generate comprehensive evaluation report
        """
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)

        report = f"""
SIGN LANGUAGE DIGITS RECOGNITION - MODEL EVALUATION REPORT
=========================================================

Model: {model_name}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{"Training Time: " + str(training_time) + " seconds" if training_time else ""}

OVERALL PERFORMANCE METRICS
---------------------------
Accuracy: {metrics['accuracy']:.4f}
Precision (Macro): {metrics['precision_macro']:.4f}
Precision (Micro): {metrics['precision_micro']:.4f}
Recall (Macro): {metrics['recall_macro']:.4f}
Recall (Micro): {metrics['recall_micro']:.4f}
F1-Score (Macro): {metrics['f1_macro']:.4f}
F1-Score (Micro): {metrics['f1_micro']:.4f}

PER-CLASS PERFORMANCE
--------------------
"""

        for i in range(NUM_CLASSES):
            report += f"Digit {i}:\n"
            report += f"  Precision: {metrics['per_class_precision'][i]:.4f}\n"
            report += f"  Recall: {metrics['per_class_recall'][i]:.4f}\n"
            report += f"  F1-Score: {metrics['per_class_f1'][i]:.4f}\n\n"

        if y_pred_proba is not None:
            roc_auc = self.calculate_roc_auc(y_true, y_pred_proba)[2]
            report += "\nROC-AUC SCORES\n--------------\n"
            for i in range(NUM_CLASSES):
                report += f"Digit {i}: {roc_auc[i]:.4f}\n"
            report += f"Macro Average: {roc_auc['macro']:.4f}\n"
            report += f"Micro Average: {roc_auc['micro']:.4f}\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to: {save_path}")

        print(report)
        return metrics

    def compare_models(self, model_results, save_path=None):
        """
        Compare multiple model results
        """
        comparison_df = pd.DataFrame()

        for model_name, results in model_results.items():
            metrics = results['metrics']
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1-Score (Macro)': metrics['f1_macro']
            }
            comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics_to_plot = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        axes = axes.ravel()

        for i, metric in enumerate(metrics_to_plot):
            axes[i].bar(comparison_df['Model'], comparison_df[metric])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for j, v in enumerate(comparison_df[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Also save CSV
            csv_path = save_path.replace('.png', '.csv')
            comparison_df.to_csv(csv_path, index=False)

        plt.show()

        return comparison_df