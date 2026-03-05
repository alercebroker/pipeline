# training_tf_custom.py
import os
import sys
import logging
import yaml
from datetime import datetime
from urllib.parse import urlparse

# Third-party imports
import tensorflow as tf
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.optimizers import Adam

# Local imports
from src.data.data_loader import get_tf_datasets
from src.models.CNN_model import DynamicStampModel
from src.models.CNN_model_modified import DynamicStampModelModified, MultiResStampModel
from src.models.ResNet import ResNetStampModel
from src.models.ResNetCustom import ResNetCustom
from src.models.ResNetCustomMini import ResNetCustomMini
from src.training.trainer import Trainer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Utility Functions ---

def save_yaml(args, path):
    """Saves the configuration dictionary to a YAML file."""
    with open(f"{path}/hparams.yaml", "w") as file:
        yaml.dump(args, file, sort_keys=False)


def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a nested dictionary for MLflow logging."""
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


def build_model(args, num_classes):
    """Factory function to build the appropriate model based on configuration."""
    hp = args['arch']
    model_type = hp.get('model_type', 'cnn')
    
    logging.info(f"Building model type: {model_type}")

    if model_type == 'resnet':
        return ResNetStampModel(
            dense_config=hp['dense_config'],
            dropout_rate=hp['dropout_rate'],
            use_batchnorm_metadata=hp['use_batchnorm_metadata'],
            num_classes=num_classes,
            use_metadata=hp['use_metadata'],
            resize_target=hp.get('resize_target', (224, 224)),
            fine_tune_at=hp.get('fine_tune_at', 0)
        )
    
    elif model_type == 'multires_cnn':
        # Definimos las claves esperadas. Siempre esperamos 'main' y las definidas en levels
        level_cols = args['loader'].get('level_cols', [])
        input_keys = ['main'] + level_cols
        
        logging.info(f"MultiRes Model Inputs: {input_keys}")
        
        return MultiResStampModel(
            conv_config=hp['conv_config'],
            dense_config=hp['dense_config'],
            dropout_rate=hp['dropout_rate'],
            num_classes=num_classes,
            input_keys=input_keys
        )

    elif model_type == 'resnet_custom':
        logging.info("Building Custom ResNet (Sulima-Lyczkowski 2025)...")
        return ResNetCustom(
            num_classes=num_classes,
            dropout_rate=hp['dropout_rate'],
            use_metadata=hp['use_metadata'],
            use_batchnorm_metadata=hp['use_batchnorm_metadata']
        )

    elif model_type == 'resnet_custom_mini':
        logging.info("Building Mini Custom ResNet (Sulima-Lyczkowski 2025)...")
        return ResNetCustomMini(
            num_classes=num_classes,
            dropout_rate=hp['dropout_rate'],
            use_metadata=hp['use_metadata'],
            use_batchnorm_metadata=hp['use_batchnorm_metadata']
        )

    elif model_type == 'net_reyes_mod':
        return DynamicStampModelModified(
            conv_config=hp['conv_config'],
            dense_config=hp['dense_config'],
            dropout_rate=hp['dropout_rate'],
            use_batchnorm_metadata=hp['use_batchnorm_metadata'],
            num_classes=num_classes,
            use_metadata=hp['use_metadata']
        )

    else:
        # Default CNN
        return DynamicStampModel(
            conv_config=hp['conv_config'],
            dense_config=hp['dense_config'],
            dropout_rate=hp['dropout_rate'],
            use_batchnorm_metadata=hp['use_batchnorm_metadata'],
            num_classes=num_classes,
            use_metadata=hp['use_metadata'],
        )


# --- Core Training Logic ---

def perform_training(run, args, experiment_name): 
    logging.info("Logging configuration parameters to MLflow...")
    flat_params = flatten_dict(args)
    mlflow.log_params(flat_params)
    logging.info("Parameters logged successfully.")

    # MLflow artifact setup
    artifact_uri = mlflow.get_artifact_uri()
    artifact_path = urlparse(artifact_uri).path
    
    #os.makedirs(artifact_path, exist_ok=True)
    #os.makedirs(os.path.join(artifact_path, 'deployment_checks'), exist_ok=True)
    #os.makedirs(os.path.join(artifact_path, 'training'), exist_ok=True)
    #os.makedirs(os.path.join(artifact_path, 'training', 'predictions'), exist_ok=True)
    #os.makedirs(os.path.join(artifact_path, 'monitoring'), exist_ok=True)

    args['artifact_path'] = artifact_path

    # 1. Load Data (Returns a dictionary now)
    data_pack = get_tf_datasets(
        batch_size=args['loader']['batch_size'], 
        args=args
    )
    
    # Unpack main components for clarity
    datasets = data_pack['datasets']
    ids = data_pack['identifiers']
    dict_info = data_pack['info']

    args.update({
        'order_features': dict_info['order_features'],
        'dict_mapping_classes': dict_info['dict_mapping_classes']
    })

    # 2. Build Model
    stamp_classifier = build_model(args, num_classes=len(dict_info['dict_mapping_classes']))

    # 3. Define Loss & Optimizer
    if args['training'].get('use_focal', False):
        loss_object = SparseCategoricalFocalLoss(gamma=2, from_logits=True)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = Adam(
        learning_rate=args['training']['lr'], 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=False
    )

    # 4. Save Configuration
    save_yaml(args, artifact_path)
    mlflow.log_artifact(os.path.join(artifact_path, "hparams.yaml"))

    # 5. Initialize Trainer
    # Note: We access data from the dictionary, making parameters explicit
    trainer = Trainer(
        model=stamp_classifier,
        loss_object=loss_object,
        optimizer=optimizer,
        args=args,
        
        train_ds=datasets['train'],
        train_ds_for_eval=datasets['train_eval'],
        val_ds=datasets['val'],
        test_ds=datasets['test'],
        test_ds_sim=datasets['test_sim'],
        
        oids_train=ids['oids']['train'],
        oids_val=ids['oids']['val'],
        oids_test=ids['oids']['test'],
        oid_test_sim=ids['oids']['test_sim'],
        
        candid_train=ids['candid']['train'],
        candid_val=ids['candid']['val'],
        candid_test=ids['candid']['test'],
        
        test_files_ids=ids['files']['test_sim'],
        
        artifact_path=artifact_path,
        dict_info=dict_info
    )

    # 6. Execute Training
    trainer.fit() 
    trainer.finalize_and_save_results()


# --- Main Entry Point ---

@hydra.main(
    config_path=os.getenv("HYDRA_CONFIG_PATH", "./configs"),
    config_name=os.getenv("HYDRA_CONFIG_NAME", "cnn_config_v1_real_rubin_allstamps_trainSN_mixed_valSN_real_modified_normIgnacio.yaml"), 
    version_base=None
)
def run(config: DictConfig) -> None:
    args = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    args = args['stamp_classifier']

    # Extract general run parameters
    name_dataset = args["name_dataset_version"]
    list_folds = args.pop('list_folds')
    results_dir = args["results_dir"]

    # Setup MLflow
    mlflow.set_tracking_uri(f"file:{results_dir}/ml-runs")
    experiment_phase = "hp_tuning" if args['is_searching_hyperparameters'] else "testing"
    experiment_name = f"classification/{name_dataset}/{experiment_phase}"
    mlflow.set_experiment(experiment_name)
    
    args_original = args.copy()
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=f"{run_name}") as parent_run:
        for fold in list_folds:
            logging.info(f'Starting Classification Stage - Fold {fold}')
            
            args = args_original.copy()
            args["loader"]['fold'] = fold
            
            with mlflow.start_run(run_name=f"Fold_{fold}_{run_name}", nested=True) as child_run:
                perform_training(child_run, args, experiment_name)


if __name__ == "__main__":
    run()