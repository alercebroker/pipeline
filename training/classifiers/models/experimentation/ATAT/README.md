# ATAT-ALeRCE

ATAT: Astronomical Transformer for time series And Tabular data consists of two Transformer models that encode light curves and features using novel Time Modulation (TM) and Quantile Feature Tokenizer (QFT) mechanisms, respectively.

<img src="https://arxiv.org/html/2405.03078v2/x1.png" alt="Description of the image" style="background-color: white; padding: 10px;">

## Main code scripts

```
📦ATAT-Refactorized

# Configurations used in training
 ┣ 📂 configs: Contains the hyperparameter configurations.
 ┃ ┗ 📜 training.yaml: A dictionary specifying the components (LC, MD, Feat, MTA, QT) to be used by ATAT.

# Data
 ┣ 📂 data:
 ┃ ┣ 📂 ztf_forced_photometry:
 ┃ ┃ ┣ 📂 processed: Contains the final dataset, and a dictionary with data information.
 ┃ ┃ ┃ ┣ 📂 [name of the dataset file]:
 ┃ ┃ ┃ ┃ ┣ 📜 dataset.h5
 ┃ ┃ ┃ ┃ ┗ 📜 dict_info.yaml: Contains critical information about dataset creation used during training.
 ┃ ┣ 📂 ELASTICC_2: (The Extended LSST Astronomical Time-Series Classification Challenge) **[WORK IN PROCESS]**
 ┃ ┃ ┣ 📂 processed 
 ┣ 📜 data_processor.py: Converts preprocessed ZTF_ff data (AstroObjects) into the model's input.

# Visualization
 ┣ 📂 notebooks **[WORK IN PROCESS]**
 ┃ ┣ 📂 data_exploration 
 ┃ ┣ 📂 meta_model 
 ┃ ┣ 📂 model_exploration
 ┃ ┗ 📂 vis_results 

# Training
 ┣ 📂 src: 
 ┃ ┣ 📂 data:
 ┃ ┃ ┣ 📂 handlers: Contains the dataloaders for the training process.
 ┃ ┃ ┣ 📂 modules: Contains the Lightning data modules for training.
 ┃ ┃ ┣ 📂 processing: Contains processing scripts to build the dataset.h5.
 ┃ ┣ 📂 layers 
 ┃ ┃ ┣ 📂 classifiers: Contains the lc, tab, and mix classifiers.
 ┃ ┃ ┣ 📂 embeddings: Embeddings used for tabular data.
 ┃ ┃ ┣ 📂 timeEncoders: Contains positional encodings and the mixture between magnitudes and times (LC transformer input).
 ┃ ┃ ┣ 📂 tokenEmbeddings: Contains the cls token.
 ┃ ┃ ┣ 📂 transformer: Contains the attention blocks and multi-head attention.
 ┃ ┃ ┃ ┣ 📂 mha
 ┃ ┃ ┣ 📂 utils
 ┃ ┃ ┗ 📜 ATAT.py: Contains the ATAT model.
 ┃ ┣ 📂 models: Contains the Lightning module for training.
 ┃ ┣ 📂 training: Contains utility functions for the training process.
 ┃ ┗ 📂 utils: Contains general utility functions.

# Run the Training and Inference
 ┣ 📜 custom_parser.py
 ┣ 📜 training.py
 ┣ 📜 inference.py 
 ┣ 📜 schema.py: Contains the names of the features used in staging.
 ┗ 📜 utils.py: Provides utility functions for data handling, MLflow experiment management, YAML file operations, classification metrics calculation, and customizable confusion matrix visualization.
 ```

## Environment Setup

Firstly, you should create the enviroment:

- conda create -n ATAT python==3.10.10
- conda activate ATAT
- pip install -r requirements.txt

## Data Acquisition 

To train ATAT, you need to obtain the preprocessed data and partitions from the [`data_acquisition folder`](../../../data_acquisition). . Detailed explanations are provided in the [`README.MD`](../../../data_acquisition/README.MD). In general, the following data is required to prepare the model's input:

1. Pickle files containing dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py) with the following keys: `'detections'`, `'non_detections'`, and `'forced_photometry'`.

2. Pickle files containing dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py) with the `'features'`.

3. Parquet files containing partitions.

## Preparing Model Input

Run the following script to extract data from the `AstroObject` files and prepare the model's input. Ensure the path is correctly set at the end of the script:

```
python data_processor.py
```

This will generate the following files in `./data/ztf_forced_photometry/processed`
* `dataset.h5`
* `dict_info.yaml`

These files are used to train the model.

## Training

To train the model, use the following commands based on the desired training components. The output will be stored in a `results` folder.

Configure the components in `configs/training.yaml`. For example, specify the hyperparameter `--experiment_type_general` as shown below:

```
# Train only with light curves (LC + MTA)
python training.py --experiment_type_general lc_mta

.
.
.

# Train with light curves and tabular information together (LC + MD + Feat + MTA)
python training.py --experiment_type_general lc_md_feat_mta
```

You could use `custom_parser.py` to configure the model hyperparameters.

## Tracking Experiments

Monitor your experiments in real-time using the MLflow interface. You can track any metric you wish, including learning curves, hyperparameters used, and, if desired, TensorBoard logs in the artifacts folder:

```
mlflow ui --backend-store-uri=file:./results/ml-runs --port 7000
```

## Inference

Inference is performed automatically at the end of the training process. The script generates a classification report, evaluates performance metrics across different days, and provides confusion matrices in the MLflow artifact folder.

If you wish to perform inference separately after training, run the following script. Ensure the path is correctly set at the end of the script:

```
python inference.py
```

For additional support, refer to the documentation or contact the ALeRCE team.
