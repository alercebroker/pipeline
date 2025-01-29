import os
import glob
import torch
import yaml
import mlflow
import logging
import pandas as pd
import numpy as np
import lightning as L

from typing import Optional
from lightning.pytorch import LightningDataModule, LightningModule

from src.models.LitATAT import LitATAT
from src.data.modules.LitData import LitData
from src.utils.ClassOrder import ClassOrder
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(dataset: LightningDataModule, 
            loaded_model: LightningModule, 
            args: yaml,
            path_save_metrics: Optional[str] = None):

    trainer = L.Trainer(logger=None)
    batches_output = trainer.predict(loaded_model, dataloaders=dataset.predict_dataloader())

    # Handling output
    path_data = os.path.join(args["general"]["data_root"], 'dict_info.yaml')
    mapping_classes = load_yaml(path_data)['mapping_classes']
    dict_mapping_classes = {value: key for key, value in mapping_classes.items()}
    sort_name_classes = list(sort_dict_by_value(dict_mapping_classes).values())
    df_list = [batch_to_df(batch, sort_name_classes) for batch in batches_output]
    df_windows_proba = pd.concat(df_list, ignore_index=True)
    df_windows_proba['id'] = df_windows_proba['id'].apply(
        lambda row: row.decode('utf-8').split('_')[0] 
        if isinstance(row, bytes) else row.split('_')[0]
        )
    #print('df_windows_proba:\n', df_windows_proba)
    #exit()
    df_lcs_proba = df_windows_proba.groupby('id').mean().reset_index()

    # Probabilities by windows
    df_windows_proba['y_pred'] = df_windows_proba['y_pred'].replace(dict_mapping_classes)
    df_windows_proba['y_true'] = df_windows_proba['y_true'].replace(dict_mapping_classes)

    # Probabilities by lightcurves (average windows probabilities)
    df_lcs_proba['y_pred'] = df_lcs_proba[sort_name_classes].idxmax(axis=1)
    df_lcs_proba['y_true'] = df_lcs_proba['y_true'].replace(dict_mapping_classes)

    # Metrics
    dict_metrics = dict()
    dict_metrics['Windows'] = calculate_metrics(y_true=df_windows_proba['y_true'],
                                                y_pred=df_windows_proba['y_pred'])
    dict_metrics['LCs'] = calculate_metrics(y_true=df_lcs_proba['y_true'],
                                            y_pred=df_lcs_proba['y_pred'])

    # Save metrics
    if path_save_metrics is not None:
        with open(f'{path_save_metrics}/classification_report_windows.txt', 'w') as file:
            file.write(dict_metrics['Windows'])
        with open(f'{path_save_metrics}/classification_report_avg_windows.txt', 'w') as file:
            file.write(dict_metrics['LCs'])

        # Save confusion matrix
        order_classes = ClassOrder.get_order(name_dataset=args["general"]["name_dataset"])

        single_confusion_matrix(y_true=df_windows_proba['y_true'], 
                                y_pred=df_windows_proba['y_pred'], 
                                order_classes=order_classes, 
                                path_save=f'{path_save_metrics}/confusion_matrix_windows.png')

        single_confusion_matrix(y_true=df_lcs_proba['y_true'], 
                                y_pred=df_lcs_proba['y_pred'], 
                                order_classes=order_classes, 
                                path_save=f'{path_save_metrics}/confusion_matrix_avg_windows.png')

    return dict_metrics


if __name__ == "__main__":

    config = {
        'mlflow_dir': 'ml-runs',

        'checkpoint': {
            'exp_name': 'classification/ztf_ff/testing',
            'run_name': '2025-01-10_02-16-55',
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
    pl_datal.setup(stage='test')

    # Model
    loaded_model = LitATAT.load_from_checkpoint(ckpt_model, map_location=device).eval()
    dict_metrics = predict(pl_datal, loaded_model, hparams)

    print(f"Windows:\n{dict_metrics['Windows']}")
    print(f"Avg windows:\n{dict_metrics['LCs']}")
