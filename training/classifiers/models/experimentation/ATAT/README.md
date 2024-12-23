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

To train ATAT, you need to obtain the preprocessed data from the [`data_acquisition folder`](https://github.com/alercebroker/pipeline/tree/main/training/classifiers/data_acquisition). In general, the following data is required to prepare the model's input:

* **Pickle files to load instances of the `AstroObject` class:** ALeRCE classifiers use the `AstroObject` class to handle data, including features, stamps, light curves, and more. AstroObject instances are stored and loaded from pickle files, which contain a list of dictionaries. Each dictionary corresponds to one astronomical object and includes the following keys: `'metadata'`, `'detections'`, `'non_detections'`, `'forced_photometry'`, `'xmatch'`, `'stamps'`, `'features'`, and `'predictions'`. These dictionaries are divided into chunks and stored across multiple pickle files. You will need two types of files:

1. **Without features**: These files contain the dataset with light curves, i.e., `'detections'`, `'non_detections'`, and `'forced_photometry'` information. For example: `data_241209_ao/astro_objects_batch_000.pkl`

2. **With features**: These files contain the `'metadata'`, and `'features'` information. The number of days used to calculate the features is indicated at the beginning of the file name. For example: `data_241209_ao_shorten_features/{days}_astro_objects_batch_000.pkl`. These files also include light curve data; however, the **Modified Julian Date (MJD)** has been altered during the feature calculation process.

* **parquet files containing partitions**. We use K-fold cross-validation to ensure consistency across train-validation-test sets. A parquet file named partitions.parquet contains the following columns:
    * `'oid'`: Object ID
    * `'alerceclass'`: Object label
    * `'ra', 'dec'`: Right Ascension and Declination
    * `'partition'`: Indicates the data split, which can be one of the following`:
        * `'test'`: The object is never used for training or validation.
        * `'training_i'`: The object is used for training in the i-th fold.
        * `'validation_i'`: The object is used for validation in the i-th fold.

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
