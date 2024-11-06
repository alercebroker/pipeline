import warnings
import logging
import colorlog
import pickle
import optuna
import yaml
import glob
import os

warnings.filterwarnings("ignore")

from custom_parser import parse_model_args, handler_parser

from src.data.modules.LitData import LitData
from src.models.LitATAT import LitATAT
from src.layers import ATAT

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer


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
    handlers=[logging.FileHandler(LOG_FILENAME, encoding="utf-8"), handler,],
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


def get_path_results(exp_path, args):
    components = []
    if args["general"]["use_lightcurves"]:
        components.append("LC")
    if args["general"]["use_metadata"]:
        components.append("MD")
    if args["general"]["use_features"]:
        components.append("FEAT")

    if components:
        exp_path += "/" + "_".join(components)
    exp_path += "/{}".format(args["general"]["experiment_name"])

    # print('exp_path: '.format(exp_path))
    return exp_path


# create folder if not exist
def handler_dirs(path, args):
    dict_name = {
        "ztf_ff": "/ZTF_ff",
        "elasticc_1": "/ELASTICC_1",
        "elasticc_2": "/ELASTICC_2",
    }

    # create child folders based on configs
    exp_path = path
    exp_path += dict_name[args["general"]["name_dataset"]]
    exp_path = get_path_results(exp_path, args)

    if args["general"]["online_opt_tt"]:
        exp_path += "/MTA"

    if args["general"]["use_augmented_dataset"]:
        exp_path += "/AUG"

    # child path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    return exp_path


def save_yaml(args, path):
    with open(f"{path}/args.yaml", "w") as file:
        yaml.dump(args, file, sort_keys=False)


def load_yaml(path):
    with open(path, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args


def handler_ckpt_path(path):
    out_path = glob.glob(path + "*.ckpt")[0]
    return out_path


def get_sampler(number_combinations, num_trials):
    print(f'The total number of combinations is {number_combinations}')
    # Determine the appropriate sampler to use
    if number_combinations < num_trials:
        # Use BruteForceSampler if the number of combinations is manageable
        sampler = optuna.samplers.BruteForceSampler()
        trials_to_run = number_combinations  # Since BruteForceSampler exhausts all combinations
    else:
        # Default to a TPE sampler or another of your choice
        sampler = None
        trials_to_run = num_trials  # Use the specified number of trials
    return sampler, trials_to_run

def count_combinations(config):
    total_combinations = 1
    for key, value in config.items():
        if isinstance(value, dict):
            total_combinations *= count_combinations(value)
        elif isinstance(value, list):
            total_combinations *= len(value)
    return total_combinations

def suggest_params(trial, config, prefix=''):
    params = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key  # Create a unique key name
        if isinstance(value, dict):
            params.update(suggest_params(trial, value, full_key))
        elif isinstance(value, list):
            params[full_key] = trial.suggest_categorical(full_key, value)
    return params

def reconstruct_nested_config(flat_params):
    nested_config = {}
    for flat_key, value in flat_params.items():
        keys = flat_key.split('.')
        temp = nested_config
        for key in keys[:-1]:
            if key not in temp:
                temp[key] = {}
            temp = temp[key]
        temp[keys[-1]] = value
    return nested_config    


def objective(trial: optuna.Trial, config: dict, list_folds: list, name_dataset: str) -> float:
    print('-'*100)
    print('1. config:\n', config)
    params = suggest_params(trial, config)
    nested_params = reconstruct_nested_config(params)
    config = OmegaConf.merge(config, OmegaConf.create(nested_params))

    config['hp_tuning'] = f"{config['imgs_params']['norm_name']}_" \
                          f"{config['imgs_params']['fig_params']['figsize'][0]}_" \
                          f"m{config['imgs_params']['fig_params']['markersize']}_" \
                          f"l{config['imgs_params']['fig_params']['linewidth']}_" \
                          f"e{config['imgs_params']['use_err']}_" \
                          f"{config['imgs_params']['input_type']}"

    print('-'*100)
    print('2. config:\n', config)
    print('-'*100)

    # Setup MLflow
    mlflow.set_tracking_uri(f"file:{config['results_dir']}/ml-runs")
    experiment_name = f"ft_classification/{name_dataset}/hyperparameter_tuning"
    mlflow.set_experiment(experiment_name)

    # Start parent run for the trial
    with mlflow.start_run(run_name=f"{config['run_name']}_trial_{trial.number}") as parent_run:

        metric_scores = {}

        for fold in list_folds:
            config['loader']['fold'] = fold

            # Start child run for each fold
            with mlflow.start_run(run_name=f"Fold_{fold}", nested=True) as child_run:

                experiment_id = child_run.info.experiment_id
                run_id = child_run.info.run_id
                EXPDIR = '{}/ml-runs/{}/{}/artifacts'.format(config['results_dir'], experiment_id, run_id)
                os.makedirs(EXPDIR, exist_ok=True)
                logging.info(f'📁 Experiment directory created: {EXPDIR}')

                # Data loading
                dataset = load_dataset(name_dataset, config)

                # Model setup
                logging.info('🗂️  Creating the model.')
                model = load_model(dataset_config=dataset.dataset_config, config=config)

                # Save params:
                if config['checkpoint']['use']:
                    logging.info('🔄 Checkpoint loading is enabled.')
                    ckpt_dir = handle_ckpt_dir(config, fold=config['loader']['fold'])
                    ckpt_model = sorted(glob.glob(ckpt_dir + "/*.ckpt"))[-1]
                    if os.path.exists(ckpt_model):
                        logging.info(f'📦 Loading checkpoint from {ckpt_model}.')
                        model = load_checkpoint(model, ckpt_model)            
                    else:
                        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_dir}")

                    loaded_config = load_yaml(path='{}/hparams.yaml'.format(ckpt_dir))
                    config['model_name'] = loaded_config['model_name']
                    logging.info(f'✅ Model parameters loaded from checkpoint: {config["model_name"]}')

                os.makedirs(f'{EXPDIR}/model', exist_ok=True)
                save_yaml(dict(model.hparams), path=f'{EXPDIR}/model/hparams.yaml')
                logging.info('💾 Model hyperparameters saved.')

                # Callbacks
                logging.info('🔧 Setting up training callbacks.')
                monitor = config['training']['monitor']
                early_stopping = EarlyStopping(
                    monitor=monitor,
                    min_delta=0.00,
                    patience=config['training']['patience'],
                    verbose=False,
                    mode="min" if 'loss' in monitor else "max",
                )
                model_summary = ModelSummary(max_depth=10, output_dir=f'{EXPDIR}/model')
                all_callbacks = [early_stopping, model_summary]

                # Loggers
                logging.info('📝 Initializing loggers for MLflow and CSV logging.')
                all_loggers = []
                mlflow_logger = MLFlowLogger(experiment_name=experiment_name,
                                             tracking_uri=mlflow.get_tracking_uri())
                mlflow_logger._run_id = child_run.info.run_id 

                csv_logger = CSVLogger(
                    save_dir=EXPDIR, 
                    name="logs",
                    version='.',
                    )

                all_loggers += [mlflow_logger, csv_logger]

                # Training
                logging.info('🏋️‍♂️ Starting model training.')
                trainer = L.Trainer(
                    callbacks=all_callbacks,
                    logger=all_loggers,
                    val_check_interval=1.0,
                    log_every_n_steps=100,
                    accelerator="gpu",
                    min_epochs=1,
                    max_epochs=config['training']['num_epochs'],
                    num_sanity_val_steps=-1,
                )
                # Train the model
                trainer.fit(model, dataset)
                logging.info('🎉 Training completed successfully.')

                # Append scores for each metric available in callback_metrics
                for key, value in trainer.callback_metrics.items():
                    if key not in metric_scores:
                        metric_scores[key] = []
                    metric_scores[key].append(value.item())
                
        # Calculate mean scores and log them
        for metric_name, scores in metric_scores.items():
            mean_score = np.mean(scores)
            mlflow.log_metric(metric_name, mean_score)
            if metric_name == monitor:
                mean_score_val = copy.copy(mean_score)

        trial.report(mean_score_val, step=0)  # Report the mean score to Optuna
        if trial.should_prune():
            raise optuna.TrialPruned()

    return mean_score_val

# Absolute path for at src level package
ABS_PATH = os.path.abspath(".")

if __name__ == "__main__":
    args = vars(parse_model_args(arg_dict=None))

    config_training = load_yaml(path="configs/training.yaml")
    if args["experiment_type_general"] not in config_training.keys():
        raise "You put an experiment name that is not found in the ./configs/training.yaml file"

    args.update(config_training[args["experiment_type_general"]])

    args = handler_parser(args)
    path = handler_dirs(ABS_PATH + f"/results/", args)
    save_yaml(args, path)

    # general updates
    args_general = args["general"]

    # Optuna setup
    number_combinations = count_combinations(config)
    sampler, trials_to_run = get_sampler(number_combinations, config['num_trials'])
    study = optuna.create_study(study_name=config['study_name'],
                                storage=config['storage'],
                                direction='maximize', 
                                sampler=sampler,
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, config, list_folds, name_dataset), n_trials=trials_to_run)

    # Extracting best parameters from optuna
    best_params_dict = study.best_trial.params
    best_params_dict = reconstruct_nested_config(best_params_dict)
    updated_config = OmegaConf.merge(config, OmegaConf.create(best_params_dict))
    updated_config = OmegaConf.to_container(updated_config, resolve=True, throw_on_missing=True)


    # Setup MLflow and retrieve the best model directly
    mlflow.set_tracking_uri(f"file:{config['results_dir']}/ml-runs")
    experiment_name = f"ft_classification/{name_dataset}/stagging"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{updated_config['run_name']}_best_model") as parent_run:

        for fold in list_folds:
            logging.info(f'📂 Starting FT Classification for Fold {fold} using the best hyperparameters.')
            updated_config['loader']['fold'] = fold

            # Data
            dataset = load_dataset(name_dataset, updated_config)

            with mlflow.start_run(run_name=f"Fold_{fold}_{updated_config['run_name']}", nested=True) as child_run:
                perform_ft_classification(child_run, updated_config, dataset, experiment_name)

    ############################  DATALOADERS  ############################
    pl_datal = LitData(**args_general)

    ############################  CALLBACKS  ############################
    all_callbacks = []
    all_callbacks += [
        ModelCheckpoint(
            monitor="loss_validation/total",  # "mix/f1s_valid"
            dirpath=path,
            save_top_k=1,
            mode="min",  # )]
            every_n_train_steps=1,
            filename="my_best_checkpoint-{step}",
        )
    ]

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

    all_loggers = []
    all_loggers += [TensorBoardLogger(save_dir=path, name="tensorboard", version=".")]
    all_loggers += [CSVLogger(save_dir=path, name=".", version=".")]

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

    trainer = Trainer(
        callbacks=all_callbacks,
        logger=all_loggers,
        val_check_interval=1.0,
        log_every_n_steps=100,
        accelerator="gpu",
        min_epochs=1,
        max_epochs=args_general["num_epochs"],
        gradient_clip_val=1.0 if pl_model.gradient_clip_val else 0.0,
        num_sanity_val_steps=-1,
    )

    # Trainer model pl routine # trsainer fit models
    trainer.fit(pl_model, pl_datal)
