import tensorflow as tf
import logging
import mlflow
import hydra
import yaml
import os 
import sys

from datetime import datetime
from urllib.parse import urlparse
from omegaconf import DictConfig, OmegaConf
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.optimizers import Adam

from src.data.data_loader import get_tf_datasets
from src.models.CNN_model import DynamicStampModel
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def save_yaml(args, path):
    with open(f"{path}/hparams.yaml", "w") as file:
        yaml.dump(args, file, sort_keys=False)

def load_yaml(path):
    with open(path, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def flatten_dict(d, parent_key='', sep='.'):
    """ Flattens a nested dictionary. """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def perform_training(run, args, experiment_name): 

    logging.info("Logging configuration parameters to MLflow...")
    flat_params = flatten_dict(args)
    mlflow.log_params(flat_params)
    logging.info("Parameters logged successfully.")

    # MLflow artifact directory
    artifact_uri = mlflow.get_artifact_uri()
    artifact_path = urlparse(artifact_uri).path  # ruta real del sistema de archivos
    os.makedirs(artifact_path, exist_ok=True)

    # Prepare datasets (placeholder: implement get_tf_datasets)
    args['artifact_path'] = artifact_path
    train_ds, train_ds_for_eval, val_ds, test_ds, oids_test, dict_info = get_tf_datasets(
        batch_size=args['loader']['batch_size'], args=args
    )
    args.update({
        'order_features': dict_info['order_features'],
        'dict_mapping_classes': dict_info['dict_mapping_classes']
        })

    # Build model
    hp = args['arch']
    stamp_classifier = DynamicStampModel(
        conv_config=hp['conv_config'],
        dense_config=hp['dense_config'],
        dropout_rate=hp['dropout_rate'],
        use_batchnorm_metadata=hp['use_batchnorm_metadata'],
        num_classes=len(dict_info['dict_mapping_classes']),
        use_metadata=hp['use_metadata']
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    if args['training'].get('use_focal', False):
        loss_object = SparseCategoricalFocalLoss(gamma=2, from_logits=False)

    optimizer = Adam(
        learning_rate=args['training']['lr'], beta_1=0.9, beta_2=0.999, amsgrad=False
        )

    # Guardar config
    save_yaml(args, artifact_path)
    mlflow.log_artifact(os.path.join(artifact_path, "hparams.yaml"))

    trainer = Trainer(
        model=stamp_classifier,
        loss_object=loss_object,
        optimizer=optimizer,
        args=args,
        train_ds=train_ds,
        train_ds_for_eval=train_ds_for_eval,
        val_ds=val_ds,
        test_ds=test_ds,
        oids_test=oids_test,
        artifact_path=artifact_path,
        dict_info=dict_info
    )

    trainer.fit()
    trainer.evaluate_and_save()


@hydra.main(config_path=os.getenv("HYDRA_CONFIG_PATH", "./configs"),
            config_name=os.getenv("HYDRA_CONFIG_NAME", "cnn_config_real_rubin_allstamps_nosatellite_clean"), 
            version_base=None)

def run(config: DictConfig) -> None:
    args = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    args = args['stamp_classifier']

    # general updates
    name_dataset = args["name_dataset_version"]
    list_folds = args.pop('list_folds')

    # Setup MLflow
    results_dir = args["results_dir"]
    mlflow.set_tracking_uri(f"file:{results_dir}/ml-runs")
    experiment_phase = "hp_tuning" if args['is_searching_hyperparameters'] else "testing"
    experiment_name = f"classification/{name_dataset}/{experiment_phase}"
    mlflow.set_experiment(experiment_name)
    
    args_original = args.copy()

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with mlflow.start_run(run_name=f"{run_name}") as parent_run:

        for fold in list_folds:
            logging.info('We are starting the Classification stage...')
            args = args_original.copy()
            args["loader"]['fold'] = fold
            with mlflow.start_run(run_name=f"Fold_{fold}_{run_name}", nested=True) as child_run:

                perform_training(child_run, args, experiment_name)


if __name__ == "__main__":
    run()
