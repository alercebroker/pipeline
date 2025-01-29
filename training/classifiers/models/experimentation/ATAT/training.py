import warnings
import logging
import colorlog
import pickle
import mlflow
import yaml
import json
import glob

import os

warnings.filterwarnings("ignore")

from datetime import datetime
from custom_parser import parse_model_args, handler_parser

from src.data.modules.LitData import LitData
from src.models.LitATAT import LitATAT
from src.layers import ATAT

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import lightning as L

from inference import predict

LOG_FILENAME = "atatRefactory.log"

# logger
logger = logging.getLogger()
logging.root.handlers = []
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s")
)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[handler,],
)

import torch
from collections import OrderedDict


def handler_checkpoint(path, args):
    od_ = OrderedDict()
    logging.info("Loading model from checkpoint ...")

    checkpoint_ = torch.load(path, map_location=torch.device("cuda"))
    for key in checkpoint_["state_dict"].keys():
        od_[key.replace("atat.", "")] = checkpoint_["state_dict"][key]
    logging.info("New keys formated ...")
    model = ATAT(**args)
    logging.info("Build ATAT  model")
    try:
        model.load_state_dict(od_, strict=False)
        logging.info("All keys matched")
    except RuntimeError as e:
        logging.error(f"Error loading model state dict: {e}")

    return model

def save_yaml(args, path):
    with open(f"{path}/hparams.yaml", "w") as file:
        yaml.dump(args, file, sort_keys=False)

def load_yaml(path):
    with open(path, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def handler_ckpt_path(path):
    out_path = glob.glob(path + "*.ckpt")[0]
    return out_path

def perform_training(run, args, experiment_name): # , pl_datal
    experiment_id = run.info.experiment_id
    run_id = run.info.run_id
    EXPDIR = 'results/ml-runs/{}/{}/artifacts'.format(experiment_id, run_id)
    os.makedirs(EXPDIR, exist_ok=True)

    args_general = args["general"]

    pl_datal = LitData(path_results=EXPDIR, **args_general)

    ############################  CALLBACKS  ############################
    all_callbacks = []
    checkpoint = ModelCheckpoint(
            monitor="loss_validation/total",  # "mix/f1s_valid"
            dirpath=EXPDIR,
            save_top_k=1,
            mode="min",  # )]
            every_n_train_steps=1,
            filename="my_best_checkpoint-{step}",
        )
    all_callbacks += [checkpoint]
    all_callbacks += [
        EarlyStopping(
            monitor="loss_validation/total", # "mix/f1s_valid"
            min_delta=0.00,
            patience=args_general["patience"],
            verbose=False,
            mode="min",
        )
    ]
    all_callbacks += [LearningRateMonitor(logging_interval="step")]

    # Loggers
    all_loggers = []
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=f"file:results/ml-runs",
        )
    mlflow_logger._run_id = run.info.run_id
    all_loggers += [mlflow_logger]
    all_loggers += [TensorBoardLogger(save_dir=EXPDIR, name="tensorboard", version="")]
    all_loggers += [CSVLogger(save_dir=EXPDIR, name=".", version="")]

    # load from checkpoint if there is one

    ############################  MODEL  ############################
    pl_model = LitATAT(**args)

    if args_general["load_pretrained_model"]:
        pl_model.atat = handler_checkpoint(
            handler_ckpt_path(args_general["src_checkpoint"]), args=args
        )

    if args_general["change_clf"]:
        pl_model.atat.change_clf(args_general["num_classes"])

    ############################  TRAINING  ############################

    if args_general['debug']:
        max_epochs=2
    else:
        max_epochs=args_general["num_epochs"]

    trainer = L.Trainer(
        callbacks=all_callbacks,
        logger=all_loggers,
        val_check_interval=1.0,
        log_every_n_steps=100,
        accelerator="gpu",
        min_epochs=1,
        max_epochs=max_epochs,
        gradient_clip_val=1.0 if pl_model.gradient_clip_val else 0.0,
        num_sanity_val_steps=-1,
    )

    # Trainer model pl routine # trsainer fit models
    trainer.fit(pl_model, pl_datal)

    # Testing
    pl_datal.setup(stage='test')
    trainer.test(dataloaders=pl_datal.test_dataloader(),
                 ckpt_path="best")
    
    # Prediction
    path_save_metrics = f'{EXPDIR}/metrics'
    os.makedirs(path_save_metrics, exist_ok=True)
    loaded_model = LitATAT.load_from_checkpoint(checkpoint.best_model_path).eval()
    _ = predict(pl_datal, loaded_model, args, path_save_metrics)
   

def run(arg_dict):
    args = vars(parse_model_args(arg_dict=arg_dict))

    config_training = load_yaml(path="configs/training.yaml")
    if args["experiment_type_general"] not in config_training.keys():
        raise "You put an experiment name that is not found in the ./configs/training.yaml file"

    args.update(config_training[args["experiment_type_general"]])
    args = handler_parser(args)

    # general updates
    args_general = args["general"]
    name_dataset = args_general["name_dataset"]

    # Handling list_folds
    list_folds = args["general"].pop('list_folds')
    try:
        list_folds = json.loads(list_folds)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al decodificar list_folds: {list_folds}") from e

    # Setup MLflow
    mlflow.set_tracking_uri(f"file:results/ml-runs")
    experiment_phase = "hp_tuning" if args_general['is_searching_hyperparameters'] else "testing"
    experiment_name = f"classification/{name_dataset}/{experiment_phase}"
    mlflow.set_experiment(experiment_name)
    
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with mlflow.start_run(run_name=f"{run_name}") as parent_run:

        for fold in list_folds:
            logging.info('We are starting the Classification stage...')

            args["general"]['fold'] = fold
            with mlflow.start_run(run_name=f"Fold_{fold}_{run_name}", nested=True) as child_run:
                perform_training(child_run, args, experiment_name) #,pl_datal


if __name__ == "__main__":

    run(arg_dict=None)
