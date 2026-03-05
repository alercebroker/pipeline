import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

def save_confusion_matrix_and_report(labels, predictions, save_dir, class_names):
    # Original confusion matrix
    cm = confusion_matrix(labels, predictions, labels=class_names)
    _plot_confusion_matrix(cm, class_names, "Confusion Matrix", os.path.join(save_dir, "confusion_matrix.png"))

    # Normalized confusion matrix (row-wise)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaNs in case of division by zero
    _plot_confusion_matrix(cm_normalized, class_names, "Normalized Confusion Matrix",
                           os.path.join(save_dir, "confusion_matrix_normalized.png"),
                           fmt=".2f")

    # Classification report
    report = classification_report(labels, predictions, target_names=class_names)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)


def log_metrics_to_mlflow(labels, predictions, class_names):
    # 1. Reporte de Clasificación
    report = classification_report(labels, predictions, target_names=class_names)
    mlflow.log_text(report, "training/metrics/classification_report.txt")

    # 2. Matriz de Confusión Original
    cm = confusion_matrix(labels, predictions, labels=class_names)
    fig_cm = _plot_confusion_matrix(cm, class_names, "Confusion Matrix")
    mlflow.log_figure(fig_cm, "training/metrics/confusion_matrix.png")
    plt.close(fig_cm) # Importante cerrar para liberar memoria
    
    # 3. Matriz de Confusión Normalizada
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig_norm = _plot_confusion_matrix(np.nan_to_num(cm_norm), class_names, "Normalized CM", fmt=".2f")
    mlflow.log_figure(fig_norm, "training/metrics/confusion_matrix_normalized.png")
    plt.close(fig_norm)

def _plot_confusion_matrix(cm, class_names, title, fmt='d'): # <--- Eliminado 'filepath'
    num_classes = len(class_names)
    cell_size = 0.8
    plt.figure(figsize=(max(6, cell_size * num_classes), max(5, cell_size * num_classes)))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})

    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    
    return plt.gcf()

#def _plot_confusion_matrix(cm, class_names, title, filepath, fmt='d'):
#    num_classes = len(class_names)
#
#    cell_size = 0.8
#    width = max(6, cell_size * num_classes)
#    height = max(5, cell_size * num_classes)
#    plt.figure(figsize=(width, height))
#
#    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
#                xticklabels=class_names, yticklabels=class_names,
#                annot_kws={"size": 14})
#
#    plt.xlabel('Predicted Label', fontsize=16)
#    plt.ylabel('True Label', fontsize=16)
#    plt.title(title, fontsize=18)
#    plt.xticks(ha='right', fontsize=12, rotation='vertical')
#    plt.yticks(fontsize=12, rotation='horizontal')
#    plt.tight_layout()
#    plt.savefig(filepath)
#    plt.close()
