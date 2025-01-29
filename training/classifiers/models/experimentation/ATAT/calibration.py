import os
import glob
import torch
import yaml
import mlflow
import logging
import pandas as pd
import numpy as np
import lightning as L
import matplotlib.pyplot as plt

from typing import Optional
from lightning.pytorch import LightningDataModule, LightningModule
from torch.nn.functional import softmax, cross_entropy
from torch.optim import LBFGS

from src.models.LitATAT import LitATAT
from src.data.modules.LitData import LitData
from src.utils.ClassOrder import ClassOrder
from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grid_search_temperature(logits, labels, T_values):
    """
    Finds the optimal temperature using grid search.

    Args:
        logits (torch.Tensor): Logits from the model (shape: [num_samples, num_classes]).
        labels (torch.Tensor): True labels (shape: [num_samples]).
        T_values (list or np.array): Candidate values for temperature.

    Returns:
        float: Optimal temperature T.
    """
    best_T = None
    best_loss = float('inf')

    for T in T_values:
        scaled_logits = logits / T
        probabilities = torch.softmax(scaled_logits, dim=1)
        loss = cross_entropy(probabilities.log(), labels)
        if loss < best_loss:
            best_loss = loss
            best_T = T

    return best_T

def grid_search_combined_metrics(logits, labels, T_values, alpha=0.5, num_bins=15):
    """
    Finds the optimal temperature using grid search with combined metrics (NLL + ECE).

    Args:
        logits (torch.Tensor): Logits from the model (shape: [num_samples, num_classes]).
        labels (torch.Tensor): True labels (shape: [num_samples]).
        T_values (list or np.array): Candidate values for temperature.
        alpha (float): Weight for ECE in the combined loss (0 <= alpha <= 1).
        num_bins (int): Number of bins for ECE calculation.

    Returns:
        float: Optimal temperature T.
        dict: Losses for NLL, ECE, and combined loss at the best T.
    """
    def compute_ece(probabilities, labels, num_bins):
        """
        Computes Expected Calibration Error (ECE).
        """
        confidences, predictions = probabilities.max(dim=1)
        accuracies = (predictions == labels).float()

        ece = 0.0
        for i in range(num_bins):
            bin_lower = i / num_bins
            bin_upper = (i + 1) / num_bins
            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if mask.sum() > 0:
                bin_accuracy = accuracies[mask].mean()
                bin_confidence = confidences[mask].mean()
                ece += mask.float().mean() * torch.abs(bin_accuracy - bin_confidence)

        return ece.item()

    best_T = None
    best_combined_loss = float('inf')
    best_nll = None
    best_ece = None

    for T in T_values:
        # Scale logits and compute probabilities
        scaled_logits = logits / T
        probabilities = torch.softmax(scaled_logits, dim=1)

        # Compute NLL
        nll = cross_entropy(probabilities.log(), labels).item()

        # Compute ECE
        ece = compute_ece(probabilities, labels, num_bins)

        # Combine NLL and ECE
        combined_loss = alpha * ece + (1 - alpha) * nll

        # Update best T
        if combined_loss < best_combined_loss:
            best_combined_loss = combined_loss
            best_T = T
            best_nll = nll
            best_ece = ece

    return best_T, {"nll": best_nll, "ece": best_ece, "combined_loss": best_combined_loss}


def reliability_diagram(logits, labels, T_before, T_after, num_bins=15, save_path=None):
    """
    Plots side-by-side reliability diagrams for before and after calibration.

    Args:
        logits (torch.Tensor): Logits from the model (shape: [num_samples, num_classes]).
        labels (torch.Tensor): True labels (shape: [num_samples]).
        T_before (float): Temperature before calibration (usually 1.0).
        T_after (float): Optimal temperature after calibration.
        num_bins (int): Number of bins to divide confidence intervals.
        save_path (str): Path to save the plot. If None, displays the plot.

    Returns:
        None. Displays or saves the reliability diagram.
    """
    # Scale logits and compute probabilities
    scaled_logits_before = logits / T_before
    probabilities_before = torch.softmax(scaled_logits_before, dim=1)
    confidences_before, predictions_before = probabilities_before.max(dim=1)
    accuracies_before = (predictions_before == labels).float()

    scaled_logits_after = logits / T_after
    probabilities_after = torch.softmax(scaled_logits_after, dim=1)
    confidences_after, predictions_after = probabilities_after.max(dim=1)
    accuracies_after = (predictions_after == labels).float()

    # Bin boundaries
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Compute fraction of positives and confidence per bin for both
    def compute_bin_metrics(confidences, accuracies, bin_lowers, bin_uppers):
        bin_confidences = []
        bin_positives = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if mask.sum() > 0:
                bin_positive = accuracies[mask].mean().item()
                bin_confidence = confidences[mask].mean().item()
            else:
                bin_positive = 0.0
                bin_confidence = 0.0
            bin_confidences.append(bin_confidence)
            bin_positives.append(bin_positive)
        return bin_confidences, bin_positives

    bin_confidences_before, bin_positives_before = compute_bin_metrics(
        confidences_before, accuracies_before, bin_lowers, bin_uppers
    )
    bin_confidences_after, bin_positives_after = compute_bin_metrics(
        confidences_after, accuracies_after, bin_lowers, bin_uppers
    )

    # Plot reliability diagrams side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot before calibration
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')  # Diagonal line
    axes[0].plot(bin_confidences_before, bin_positives_before, 'o-', label='Model Calibration')
    axes[0].bar(bin_confidences_before, 
                [p - c for p, c in zip(bin_positives_before, bin_confidences_before)],
                width=0.03, alpha=0.6, label='Gap (Positives - Confidence)', color='orange')
    axes[0].set_title(f"Before Calibration (T={T_before})")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Fraction of Positive Samples")
    axes[0].legend()
    axes[0].grid(True)

    # Plot after calibration
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')  # Diagonal line
    axes[1].plot(bin_confidences_after, bin_positives_after, 'o-', label='Model Calibration')
    axes[1].bar(bin_confidences_after, 
                [p - c for p, c in zip(bin_positives_after, bin_confidences_after)],
                width=0.03, alpha=0.6, label='Gap (Positives - Confidence)', color='orange')
    axes[1].set_title(f"After Calibration (T={T_after:.3f})")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].legend()
    axes[1].grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Reliability diagrams saved to {save_path}")
        plt.close()
    else:
        plt.show()


def predicted_probabilities_histogram(logits, T_before, T_after, num_bins=15, save_path=None):
    """
    Plots side-by-side histograms of predicted probabilities before and after calibration.

    Args:
        logits (torch.Tensor): Logits from the model (shape: [num_samples, num_classes]).
        T_before (float): Temperature before calibration (usually 1.0).
        T_after (float): Optimal temperature after calibration.
        num_bins (int): Number of bins for the histogram.
        save_path (str): Path to save the plot. If None, displays the plot.

    Returns:
        None. Displays or saves the histogram.
    """
    # Scale logits and compute probabilities
    scaled_logits_before = logits / T_before
    probabilities_before = torch.softmax(scaled_logits_before, dim=1)
    confidences_before, _ = probabilities_before.max(dim=1)

    scaled_logits_after = logits / T_after
    probabilities_after = torch.softmax(scaled_logits_after, dim=1)
    confidences_after, _ = probabilities_after.max(dim=1)

    # Plot histograms side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Histogram before calibration
    axes[0].hist(confidences_before.cpu().numpy(), bins=num_bins, range=(0, 1), color='blue', alpha=0.7)
    axes[0].set_title(f"Histogram Before Calibration (T={T_before})")
    axes[0].set_xlabel("Predicted Probability (Confidence)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    # Histogram after calibration
    axes[1].hist(confidences_after.cpu().numpy(), bins=num_bins, range=(0, 1), color='green', alpha=0.7)
    axes[1].set_title(f"Histogram After Calibration (T={T_after:.3f})")
    axes[1].set_xlabel("Predicted Probability (Confidence)")
    axes[1].grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Histograms saved to {save_path}")
        plt.close()
    else:
        plt.show()


def generate_classification_report(logits, labels, T, class_names=None):
    """
    Generates a classification report after applying temperature scaling.

    Args:
        logits (torch.Tensor): Logits from the model (shape: [num_samples, num_classes]).
        labels (torch.Tensor): True labels (shape: [num_samples]).
        T (float): Optimal temperature for scaling logits.
        class_names (list, optional): Names of the classes. Defaults to None.

    Returns:
        str: Classification report as a formatted string.
    """
    # Scale logits with temperature
    scaled_logits = logits / T
    probabilities = torch.softmax(scaled_logits, dim=1)

    # Predicted labels
    predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

    # True labels
    true_labels = labels.cpu().numpy()

    # Generate classification report
    report = classification_report(true_labels, predictions, target_names=class_names, digits=4)
    return report


def predict(dataset: LightningDataModule, 
            loaded_model: LightningModule, 
            args: yaml,
            optimal_T: float):

    trainer = L.Trainer(logger=None)
    batches_output = trainer.predict(loaded_model, dataloaders=dataset.predict_dataloader())

    # Handling output
    path_data = os.path.join(args["general"]["data_root"], 'dict_info.yaml')
    mapping_classes = load_yaml(path_data)['mapping_classes']
    dict_mapping_classes = {value: key for key, value in mapping_classes.items()}
    sort_name_classes = list(sort_dict_by_value(dict_mapping_classes).values())
    df_list = [batch_to_df_logits(batch, sort_name_classes) for batch in batches_output]
    df_windows_logits = pd.concat(df_list, ignore_index=True)
    df_windows_logits['id'] = df_windows_logits['id'].apply(
        lambda row: row.decode('utf-8').split('_')[0] 
        if isinstance(row, bytes) else row.split('_')[0]
        )
    df_lcs_logits = df_windows_logits.groupby('id').mean().reset_index()

    logits = torch.tensor(np.vstack([df_lcs_logits[class_label].values for class_label in sort_name_classes]).T, device=device)
    labels = torch.tensor(df_lcs_logits['y_true'].values, dtype=torch.long, device=device)

    save_path = './results/images/calibration'
    os.makedirs(save_path, exist_ok=True)

    # Before calibration
    reliability_diagram(
        logits, labels, T_before=1.0, T_after=optimal_T, num_bins=15,
        save_path=f"{save_path}/reliability_diagrams.png"
    )
    predicted_probabilities_histogram(
        logits, T_before=1.0, T_after=optimal_T, num_bins=15,
        save_path=f"{save_path}/histograms_pred_probs.png"
    )



def calibrate(dataset: LightningDataModule, 
              loaded_model: LightningModule, 
              args: yaml):

    trainer = L.Trainer(logger=None)
    batches_output = trainer.predict(loaded_model, dataloaders=dataset.val_dataloader())

    # Handling output
    path_data = os.path.join(args["general"]["data_root"], 'dict_info.yaml')
    mapping_classes = load_yaml(path_data)['mapping_classes']
    dict_mapping_classes = {value: key for key, value in mapping_classes.items()}
    sort_name_classes = list(sort_dict_by_value(dict_mapping_classes).values())
    df_list = [batch_to_df_logits(batch, sort_name_classes) for batch in batches_output]
    df_windows_logits = pd.concat(df_list, ignore_index=True)
    df_windows_logits['id'] = df_windows_logits['id'].apply(
        lambda row: row.decode('utf-8').split('_')[0] 
        if isinstance(row, bytes) else row.split('_')[0]
        )
    df_lcs_logits = df_windows_logits.groupby('id').mean().reset_index()

    logits = torch.tensor(np.vstack([df_lcs_logits[class_label].values for class_label in sort_name_classes]).T, device=device)
    labels = torch.tensor(df_lcs_logits['y_true'].values, dtype=torch.long, device=device)

    T_values = np.linspace(0.5, 3.0, 50)
    optimal_T, dict_losses = grid_search_combined_metrics(logits, labels, T_values)

    return optimal_T, dict_losses


if __name__ == "__main__":

    config = {
        'mlflow_dir': 'ml-runs',

        'checkpoint': {
            'exp_name': 'classification/ztf_ff/testing',
            'run_name': '2025-01-20_23-11-17',
            'results_dir': 'results',
        },

        'loader': {
            'fold': 0
            }
    }

    fold = config['loader']['fold']
    ckpt_dir = handle_ckpt_dir(config, fold=fold)
    ckpt_model = sorted(glob.glob(ckpt_dir + "/*.ckpt"))[-1]

    experiment_id = get_experiment_id_mlflow(config['checkpoint']['exp_name'])
    run_name = f"Fold_{fold}_{config['checkpoint']['run_name']}"
    run_id = get_run_id_mlflow(experiment_id, run_name)
    EXPDIR = 'results/ml-runs/{}/{}/artifacts'.format(experiment_id, run_id)

    # Data
    hparams = load_yaml(f'{ckpt_dir}/hparams.yaml')
    pl_datal = LitData(
        path_results=EXPDIR,
        **hparams['general']
        )
    pl_datal.setup(stage='fit')

    # Model
    loaded_model = LitATAT.load_from_checkpoint(ckpt_model, map_location=device).eval()
    optimal_T, dict_losses = calibrate(pl_datal, loaded_model, hparams)

    logging.info(f"Optimal Temperature: {optimal_T}")
    logging.info(f"Losses: {dict_losses}")

    pl_datal.setup(stage='test')
    predict(pl_datal, loaded_model, hparams, optimal_T)
