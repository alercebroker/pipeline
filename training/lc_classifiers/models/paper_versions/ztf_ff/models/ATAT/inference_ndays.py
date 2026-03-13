import os
import gc
import glob
import torch
import yaml
import mlflow
import colorlog
import logging
import pandas as pd
import numpy as np
import copy
import lightning as L

from typing import Optional
from lightning.pytorch import LightningDataModule, LightningModule

from src.models.LitATAT import LitATAT
from src.data.modules.LitData import LitData
from src.utils.ClassOrder import ClassOrder
from utils import *

logger = logging.getLogger()
logging.root.handlers = []
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[handler,],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(dataset: LightningDataModule, 
            loaded_model: LightningModule, 
            args: yaml,
            path_save_metrics: Optional[str] = None,
            path_save_predictions: Optional[str] = None,
            stage: Optional[str] = 'test',
            time_eval: Optional[int] = None):

    logging.info(f"Starting prediction for {stage} with time_eval={time_eval}")

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
    df_lcs_proba = df_windows_proba.groupby('id').mean().reset_index()

    # Probabilities by windows
    df_windows_proba['y_pred'] = df_windows_proba['y_pred'].replace(dict_mapping_classes)
    df_windows_proba['y_true'] = df_windows_proba['y_true'].replace(dict_mapping_classes)

    # Probabilities by lightcurves (average windows probabilities)
    df_lcs_proba['y_pred'] = df_lcs_proba[sort_name_classes].idxmax(axis=1)
    df_lcs_proba['y_true'] = df_lcs_proba['y_true'].replace(dict_mapping_classes)

    if path_save_predictions is not None:
        os.makedirs(f'{path_save_predictions}/Windows', exist_ok=True)
        os.makedirs(f'{path_save_predictions}/LCs', exist_ok=True)
        df_windows_proba.to_parquet(f'{path_save_predictions}/Windows/predictions_{stage}_{time_eval}.parquet')
        df_lcs_proba.to_parquet(f'{path_save_predictions}/LCs/predictions_{stage}_{time_eval}.parquet')

    # Metrics
    dict_metrics = dict()
    dict_metrics['Windows'] = calculate_metrics(y_true=df_windows_proba['y_true'],
                                                y_pred=df_windows_proba['y_pred'])
    dict_metrics['LCs'] = calculate_metrics(y_true=df_lcs_proba['y_true'],
                                            y_pred=df_lcs_proba['y_pred'])

    # Save metrics
    if path_save_metrics is not None:
        path_save = f'{path_save_metrics}/classification_report'
        os.makedirs(path_save, exist_ok=True)
        with open(f'{path_save}/ndays{time_eval}_clf_report_avg_windows.txt', 'w') as file:
            file.write(dict_metrics['LCs'])

        # Save confusion matrix
        name_dataset = args["general"]["name_dataset"]
        if name_dataset == 'ztf_ff_20folds': name_dataset = 'ztf_ff' 
        elif name_dataset == 'ztf_ff_sanchez_tax_20folds': name_dataset = 'ztf_ff_sanchez_tax' 
        order_classes = ClassOrder.get_order(name_dataset=name_dataset)

        path_save = f'{path_save_metrics}/confusion_matrices'
        os.makedirs(path_save, exist_ok=True)
        single_confusion_matrix(y_true=df_lcs_proba['y_true'], 
                                y_pred=df_lcs_proba['y_pred'], 
                                order_classes=order_classes, 
                                path_save=f'{path_save}/ndays{time_eval}_cm_avg_windows.png')

    return dict_metrics


def get_mlflow_runs(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    runs = runs[
        (runs.status == 'FINISHED') & 
        (runs['metrics.mix/f1s_test/None_days'].notna())
        ]
    return runs

def compute_ndays_results(EXPDIR, list_time_to_eval, ckpt_model, stage='test', data_root=None):
    logging.info(f"Computing results for {stage} in {EXPDIR}")

    hparams = load_yaml(f'{EXPDIR}/hparams.yaml') 
    assert all(item in list_time_to_eval for item in hparams['general']['list_time_to_eval'])

    if data_root is not None:
        hparams['general']['data_root'] = data_root

    path_save_predictions = f'{EXPDIR}/ndays_results'
    os.makedirs(path_save_predictions, exist_ok=True)

    for time_eval in list_time_to_eval:        
        hparams['general']['list_time_to_eval'] = [time_eval]
        pl_datal = LitData(
            path_results=EXPDIR,
            **hparams['general']
            )
        
        pl_datal.setup(stage=stage)

        if time_eval is not None:
            mask_time = (pl_datal.test_dataset.time_alert <= time_eval).bool()
            pl_datal.test_dataset.mask = pl_datal.test_dataset.mask * mask_time

        # Model
        loaded_model = LitATAT.load_from_checkpoint(ckpt_model, map_location=device).eval()
        _ = predict(pl_datal, loaded_model, hparams, 
                    path_save_predictions=path_save_predictions, 
                    stage=stage,
                    time_eval=time_eval)
        
        # 3. LIMPIEZA CRUCIAL AL FINAL DE CADA ITERACIÓN
        del pl_datal
        del loaded_model
        gc.collect() # Limpia objetos en RAM (Python)
        torch.cuda.empty_cache() # Libera la caché de la GPU

    logging.info("Finished computing results for all time evaluations.")

if __name__ == "__main__":

    name_datasets = [
        'ztf_ff_20folds',
        'ztf_ff_sanchez_tax_20folds'
    ]

    for name_dataset in name_datasets:

        config = {
            'mlflow_dir': 'ml-runs',

            'checkpoint': {
                'exp_name': f'classification/{name_dataset}/testing',
                'run_name': None, #'2025-03-16_23-19-35',
                'results_dir': 'results',
            },

            #'loader': {
            #    'fold': 17,
            #    }
        }

        stage = 'test'
        list_time_to_eval = [8, 16, 32, 64, 128, 256, 512, 1024, None]

        if name_dataset == 'ztf_ff_20folds':
            data_root = f'data/processed/ds_pre250408_ndetge8_20folds_pos260202'
        elif name_dataset == 'ztf_ff_sanchez_tax_20folds':
            data_root = f'data/processed/ds_pre250408_ndetge8_sanchez_tax_20folds_pos260202'
        else:
            raise 'Dataset does not exit'


        mlflow.set_tracking_uri(f"file:{config['checkpoint']['results_dir']}/ml-runs")
        experiment_id = get_experiment_id_mlflow(config['checkpoint']['exp_name'])
        
        if config['checkpoint']['run_name'] is None:
            runs = get_mlflow_runs(experiment_id)
            for _, run in runs.iterrows():
                run_id = run['run_id']
                fold = run['params.general/fold']

                EXPDIR = 'results/ml-runs/{}/{}/artifacts'.format(experiment_id, run_id)
                ckpt_model = sorted(glob.glob(EXPDIR + "/*.ckpt"))[-1]

                compute_ndays_results(EXPDIR, list_time_to_eval, ckpt_model, stage=stage, data_root=data_root)

        else:     
            fold = config['loader']['fold']
            ckpt_dir = handle_ckpt_dir(config, fold=fold)
            ckpt_model = sorted(glob.glob(ckpt_dir + "/*.ckpt"))[-1]

            run_name = f"Fold_{fold}_{config['checkpoint']['run_name']}"
            run_id = get_run_id_mlflow(experiment_id, run_name)
            EXPDIR = 'results/ml-runs/{}/{}/artifacts'.format(experiment_id, run_id)

            compute_ndays_results(EXPDIR, list_time_to_eval, ckpt_model, stage=stage, data_root=data_root)
