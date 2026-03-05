import os
import mlflow
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from src.training.losses import balanced_xentropy

def evaluate_and_log(dataset, name, model, dict_info, artifact_path):
    y_true_all = []
    y_pred_all = []

    for (stamps, metadata), labels in dataset:
        preds = model.stamp_classifier((stamps, metadata), training=False)
        preds = tf.argmax(preds, axis=1)
        y_true_all.extend(labels.numpy())
        y_pred_all.extend(preds.numpy())

    labels_map = dict_info["dict_mapping_classes"]
    target_names = [labels_map[i] for i in sorted(labels_map)]

    # Report
    report = classification_report(y_true_all, y_pred_all, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(artifact_path, "metrics", f"{name}_classification_report.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_df.to_csv(report_path)

    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}")
    cm_path = os.path.join(artifact_path, "metrics", f"{name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Log to MLflow
    mlflow.log_artifact(report_path, artifact_path="metrics")
    mlflow.log_artifact(cm_path, artifact_path="metrics")


def eval_step(model, dataset):
    logits_list, labels_list = [], []
    for (samples, md), labels in dataset:
        # El modelo ahora devuelve logits crudos (linear)
        logits = model((samples, md), training=False)
        logits_list.append(logits)
        labels_list.append(labels)

    logits_tensor = tf.concat(logits_list, axis=0)
    labels = tf.concat(labels_list, axis=0)

    # El argmax funciona exactamente igual con logits que con probabilidades
    preds_argmax = logits_tensor.numpy().argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(), preds_argmax, average='macro', zero_division=0)
    
    # Le pasamos los logits directos porque balanced_xentropy hace el softmax por dentro
    xentropy = balanced_xentropy(labels, logits_tensor)

    # IMPORTANTE: Devolvemos preds_argmax y los logits crudos (.numpy()) al final
    return precision, recall, f1, xentropy, labels.numpy(), preds_argmax, logits_tensor.numpy()