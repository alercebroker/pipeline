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

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import glob

# (Añade esta función en algún lugar de tu script)
def get_model_path_from_old_mlflow(checkpoint_args, fold):
    """
    Finds the path to a model artifact from a previous MLflow experiment.

    Args:
        checkpoint_args (dict): Dictionary with 'results_dir', 'exp_name', 'run_name'.
        fold (int): The fold number to format the run name.

    Returns:
        tuple: A tuple containing (path_to_model_file, path_to_hparams_file).
               Returns (None, None) if not found.
    """
    old_tracking_uri = f"file:{checkpoint_args['results_dir']}/ml-runs"
    logging.info(f"Connecting to old MLflow URI to find model: {old_tracking_uri}")
    
    # 1. Create a client pointing to the old location
    client = MlflowClient(tracking_uri=old_tracking_uri)

    try:
        # 2. Find the experiment
        experiment = client.get_experiment_by_name(checkpoint_args['exp_name'])
        if not experiment:
            logging.error(f"Experiment '{checkpoint_args['exp_name']}' not found in {old_tracking_uri}")
            return None, None
        
        experiment_id = experiment.experiment_id

        # 3. Search for the specific run
        run_name_to_find = f"Fold_{fold}_{checkpoint_args['run_name']}"
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{run_name_to_find}'"
        )

        if not runs:
            logging.error(f"Run with name '{run_name_to_find}' not found in experiment '{checkpoint_args['exp_name']}'")
            return None, None
        
        # Assuming the first match is the correct one
        target_run = runs[0]
        run_id = target_run.info.run_id
        artifact_uri = target_run.info.artifact_uri
        
        # 4. Construct the file path
        # The artifact_uri for file stores is the absolute path
        artifact_path = urlparse(artifact_uri).path
        
        # Find the .keras model file and hparams.yaml
        model_files = sorted(glob.glob(os.path.join(artifact_path, "*.keras")))
        if not model_files:
            logging.error(f"No .keras model file found in artifacts for run_id {run_id}")
            return None, None
            
        model_path = model_files[-1] # Get the latest one if there are multiple
        hparams_path = os.path.join(artifact_path, "hparams.yaml")
        
        logging.info(f"Found model to load: {model_path}")
        logging.info(f"Found hparams to load: {hparams_path}")
        
        return model_path, hparams_path

    except MlflowException as e:
        logging.error(f"An MLflow error occurred while searching for the old model: {e}")
        return None, None

import yaml

def load_yaml(path):
    with open(path, 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def perform_training(run, args, experiment_name): 

    # MLflow artifact directory
    artifact_uri = mlflow.get_artifact_uri()
    artifact_path = urlparse(artifact_uri).path  # ruta real del sistema de archivos
    os.makedirs(artifact_path, exist_ok=True)

    if args["load_pretrained_model"]:
        # Usa la nueva función para obtener la ruta sin cambiar el URI global
        ckpt_model_path, hparams_path = get_model_path_from_old_mlflow(args['checkpoint'], fold=0) #fold=args["loader"]['fold'])
        
        if ckpt_model_path and hparams_path:
            logging.info(f"Loading pretrained model weights from: {ckpt_model_path}")
            # Opcional: Cargar los hparams antiguos si los necesitas
            old_hparams = load_yaml(hparams_path)
            # logging.info("Loaded hparams from old run:", old_hparams)
        else:
            logging.error("Could not find pretrained model. Training from scratch.")
            # Decide si quieres detener el proceso o continuar entrenando desde cero
            # sys.exit(1) 

    args.update({'dict_mapping_classes': old_hparams['dict_mapping_classes']})

    # Prepare datasets (placeholder: implement get_tf_datasets)
    args['artifact_path'] = artifact_path
    train_ds, train_ds_for_eval, val_ds, test_ds, oids_test, dict_info = get_tf_datasets(
        batch_size=args['loader']['batch_size'], args=args, load_pretrained_model=args["load_pretrained_model"]
    )
    args.update({
        'order_features': dict_info['order_features'],
        'dict_mapping_classes': dict_info['dict_mapping_classes'],
        })

    # Build model
    hp = old_hparams['arch']
    stamp_classifier = DynamicStampModel(
        conv_config=hp['conv_config'],
        dense_config=hp['dense_config'],
        dropout_rate=hp['dropout_rate'],
        use_batchnorm_metadata=hp['use_batchnorm_metadata'],
        num_classes=len(dict_info['dict_mapping_classes']),
        use_metadata=hp['use_metadata']
    )

    logging.info("Building model to load weights...")
    sample_input, _ = next(iter(train_ds))
    _ = stamp_classifier(sample_input, training=False) 
    stamp_classifier.load_weights(ckpt_model_path) # Usar load_weights es más seguro si la arquitectura no ha cambiado
    logging.info("Weights loaded successfully.")


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
            config_name=os.getenv("HYDRA_CONFIG_NAME", "cnn_config_dp1_to_rubin"), 
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
    
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with mlflow.start_run(run_name=f"{run_name}") as parent_run:

        for fold in list_folds:
            logging.info('We are starting the Classification stage...')

            args["loader"]['fold'] = fold
            with mlflow.start_run(run_name=f"Fold_{fold}_{run_name}", nested=True) as child_run:

                perform_training(child_run, args, experiment_name)


if __name__ == "__main__":
    run()
