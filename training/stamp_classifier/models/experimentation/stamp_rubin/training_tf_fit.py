import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import logging
import mlflow
import hydra
import yaml
import json
import os 

from datetime import datetime
from urllib.parse import urlparse
from omegaconf import DictConfig, OmegaConf
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.optimizers import Adam

from src.data.data_loader import get_tf_datasets
from src.models.CNN_model import DynamicStampModel
from src.models.base_model import CustomModel
from inference import evaluate_and_log

class MLflowLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, value in logs.items():
            mlflow.log_metric(metric_name, value, step=epoch)

def save_learning_curves(history, history_path):
    os.makedirs(history_path, exist_ok=True)

    # CSV del historial
    history_df = pd.DataFrame(history.history)
    csv_path = os.path.join(history_path, "training_history.csv")
    history_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="metrics/learning_curves")

    # PNGs de curvas y log_metric
    for epoch in range(len(history.history[list(history.history.keys())[0]])):
        for metric, values in history.history.items():
            mlflow.log_metric(metric, values[epoch], step=epoch)

    for metric, values in history.history.items():
        plt.figure()
        plt.plot(values, label=f"train {metric}")
        if f"val_{metric}" in history.history:
            plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")
        plt.title(f"{metric} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        img_path = os.path.join(history_path, f"{metric}.png")
        plt.savefig(img_path)
        plt.close()
        mlflow.log_artifact(img_path, artifact_path="metrics/learning_curves")

def save_yaml(args, path):
    with open(f"{path}/hparams.yaml", "w") as file:
        yaml.dump(args, file, sort_keys=False)

def load_yaml(path):
    with open(path, "r") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def perform_training(run, args, experiment_name): 

    # MLflow artifact directory
    artifact_uri = mlflow.get_artifact_uri()
    artifact_path = urlparse(artifact_uri).path  # ruta real del sistema de archivos
    os.makedirs(artifact_path, exist_ok=True)

    # Prepare datasets (placeholder: implement get_tf_datasets)
    args['artifact_path'] = artifact_path
    train_ds, val_ds, test_ds, dict_info = get_tf_datasets(
        batch_size=args['loader']['batch_size'], args=args
    )

    # Build model
    hp = args['arch']
    stamp_classifier = DynamicStampModel(
        conv_config=hp['conv_config'],
        dense_config=hp['dense_config'],
        dropout_rate=hp['dropout_rate'],
        use_batchnorm_metadata=hp['use_batchnorm_metadata'],
        num_classes=len(dict_info['dict_mapping_classes']),
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if args['training'].get('use_focal', False):
        loss = SparseCategoricalFocalLoss(gamma=2, from_logits=True)

    optimizer = Adam(learning_rate=args['training']['lr'])

    # Wrap
    custom_model = CustomModel(
        stamp_classifier=stamp_classifier,
        loss_object=loss,
        optimizer=optimizer
    )

    custom_model.compile(
        optimizer=optimizer,
        loss=loss,
        #metrics=["sparse_categorical_accuracy"]
    )

    # Guardar config
    save_yaml(args, artifact_path)
    mlflow.log_artifact(os.path.join(artifact_path, "hparams.yaml"))

    # Callbacks
    log_dir = os.path.join(artifact_path, "logs")
    callbacks = []
    callbacks.append(
        EarlyStopping(
            monitor=args['training']['monitor'],
            patience=args['training']['patience'],
            mode="min" if 'loss' in args['training']['monitor'] else "max",
            restore_best_weights=True
        )
    )
    callbacks.append(
        ModelCheckpoint(
            filepath=os.path.join(artifact_path, "best_model.keras"),
            monitor=args['training']['monitor'],
            save_best_only=True,
            mode="min" if 'loss' in args['training']['monitor'] else "max"
        )
    )
    callbacks.append(
        TensorBoard(log_dir=log_dir)
    )
    #callbacks.append(MLflowLoggingCallback())

    # Train
    history = custom_model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=500,
        epochs=args['training']['num_epochs'] if not args['debug'] else 2,
        callbacks=callbacks
    )

    # Guardar modelo y logs
    mlflow.log_artifact(os.path.join(artifact_path, "best_model.keras"))
    mlflow.log_artifacts(log_dir, artifact_path="logs")

    # Learning Curves
    history_path = os.path.join(artifact_path, "metrics", "learning_curves")
    save_learning_curves(history, history_path)

    # === Inference ===
    # Reload best model
    best_model_path = os.path.join(artifact_path, "best_model.keras")
    loaded_model = tf.keras.models.load_model(best_model_path)

    # Evaluate and log metrics
    evaluate_and_log(train_ds, "train", loaded_model, dict_info, artifact_path)
    evaluate_and_log(val_ds, "val", loaded_model, dict_info, artifact_path)
    evaluate_and_log(test_ds, "test", loaded_model, dict_info, artifact_path)

    print(f"Inference complete. Metrics saved to: {os.path.join(artifact_path, 'metrics')}")


@hydra.main(config_path=os.getenv("HYDRA_CONFIG_PATH", "./configs"),
            config_name=os.getenv("HYDRA_CONFIG_NAME", "cnn_config_v0"), 
            version_base=None)

def run(config: DictConfig) -> None:
    args = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    args = args['stamp_classifier']

    # general updates
    name_dataset = args["name_dataset_version"]
    list_folds = args.pop('list_folds')

    # Setup MLflow
    mlflow.set_tracking_uri(f"file:results/ml-runs")
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
