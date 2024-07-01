# ATAT-ALeRCE

ATAT: Astronomical Transformer for time series And Tabular data consists of two Transformer models that encode light curves and features using novel Time Modulation (TM) and Quantile Feature Tokenizer (QFT) mechanisms, respectively.

<img src="https://arxiv.org/html/2405.03078v2/x1.png" alt="Description of the image" style="background-color: white; padding: 10px;">

**Code authors**:
- Nicolas Astorga 
- Bastian Gamboa
- Daniel Moreno

## Main code scripts

```
📦ATAT-Refactorized

# Configurations used in training
 ┣ 📂 configs: Contains the hyperparameter configurations.
 ┃ ┗ 📜 training.yaml: A dictionary specifying the components (LC, MD, Feat, MTA, QT) to be used by ATAT.

# Data
 ┣ 📂 data:
 ┃ ┣ 📂 datasets:
 ┃ ┃ ┣ 📂 ZTF_ff: (ZTF forced photometry)
 ┃ ┃ ┃ ┣ 📂 final: Contains the final dataset, quantiles, and a dictionary with data information.
 ┃ ┃ ┃ ┃ ┣ 📂 quantiles
 ┃ ┃ ┃ ┃ ┣ 📜 dataset.h5
 ┃ ┃ ┃ ┃ ┗ 📜 dict_info.yaml: Contains critical information about dataset creation used during training.
 ┃ ┃ ┃ ┣ 📂 partitions: Contains versions of the data partitions.
 ┃ ┃ ┃ ┣ 📂 processed: 
 ┃ ┃ ┃ ┃ ┣ 📂 data_231206: Light curves in parquet format derived from raw data using extracting_ztf_md_feat.
 ┃ ┃ ┃ ┃ ┣ 📂 md_feat_231206_v2
 ┃ ┃ ┃ ┃ ┃ ┣ 📂 metadata: Metadata in parquet format derived from raw data using extracting_ztf_md_feat.
 ┃ ┃ ┃ ┃ ┃ ┣ 📂 features: Features per day in parquet format derived from raw data using extracting_ztf_md_feat.
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📂 16_days: Features calculated on day 16.
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ ...  
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📂 2048_days: Features calculated on day 2048.
 ┃ ┃ ┃ ┃ ┗ 📂 ulens_keep: Contains CSVs of the ulens that were effectively used.
 ┃ ┃ ┃ ┗ 📂 raw 
 ┃ ┃ ┃ ┃ ┣ 📂 data_231206_ao: Batched light curves of astro_objects.
 ┃ ┃ ┃ ┃ ┣ 📂 data_231206_ao_features: Batched features of astro_objects.
 ┃ ┃ ┃ ┃ ┗ 📂 pipeline: ALeRCE pipeline necessary for extracting information from astro_objects files.
 ┃ ┃ ┣ 📂 ELASTICC_2: (The Extended LSST Astronomical Time-Series Classification Challenge) **[IN PROCESS]**
 ┃ ┃ ┃ ┣ 📂 final 
 ┃ ┃ ┃ ┣ 📂 partitions
 ┃ ┃ ┃ ┣ 📂 processed 
 ┃ ┃ ┃ ┣ 📂 raw 
 ┃ ┣ 📂 src: Contains the .py files for ATAT processing, and for generating partitions and datasets. 
 ┃ ┣ 📜 ZTF_ff_processed_to_final.py: Generates the final ZTF dataset using the processed data.
 ┃ ┗ 📜 ZTF_ff_raw_to_processed.py: Converts raw ZTF data into processed form.

# Visualization
 ┣ 📂 notebooks **[IN PROCESS]**
 ┃ ┣ 📂 data_exploration 
 ┃ ┣ 📂 meta_model 
 ┃ ┣ 📂 model_exploration
 ┃ ┗ 📂 vis_results 
 ┃ ┃ ┣ 📜 classification_metrics.ipynb 
 ┃ ┃ ┣ 📜 confusion_matrices.ipynb 
 ┃ ┃ ┗ 📜 plot_metrics_times.ipynb 

# Training
 ┣ 📂 src: 
 ┃ ┣ 📂 data:
 ┃ ┃ ┣ 📂 handlers: Contains the dataloaders for the training process.
 ┃ ┃ ┣ 📂 modules: Contains the Lightning data modules for training.
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
 ┃ ┗ 📂 utils: Contains various plotting functions.

# Run the Training and Inference
 ┣ 📜 custom_parser.py
 ┣ 📜 training.py
 ┗ 📜 inference_ztf.py 
 ```

The steps to run ATAT are the following:

```
git clone https://github.com/alercebroker/pipeline.git
cd pipeline/training/lc_classifier_ztf/ATAT_ALeRCE
```

## Environment Setup

Firstly, you should create the enviroment:

- conda create -n ATAT python==3.10.10
- conda activate ATAT
- pip install -r requirements.txt

## Data Acquisition 

The raw ZTF_ff data is stored at `quimal-cpu2` in `/home/db_storage/ztf_forced_photometry`. Ensure to replicate the same file structure locally: 

```
mkdir -p ./data/datasets/ZTF_ff/raw
scp -r <user_name>@146.83.185.161:/home/db_storage/ztf_forced_photometry/data_231206_ao ./data/datasets/ZTF_ff/raw 
scp -r <user_name>@146.83.185.161:/home/db_storage/ztf_forced_photometry/data_231206_ao_features ./data/datasets/ZTF_ff/raw
```

```
mkdir -p ./data/datasets/ZTF_ff/processed
scp -r <user_name>@146.83.185.161:/home/db_storage/ztf_forced_photometry/data_231206 ./data/datasets/ZTF_ff/processed 
```

Then, run the following to obtain the processed data:

```
python data/ZTF_ff_raw_to_processed.py
```

Additionally, retrieve the mislabeled ulens and those that are to be retained:

```
scp -r <user_name>@146.83.185.161:/home/db_storage/ztf_forced_photometry/ulens_keep ./data/datasets/ZTF_ff/processed 
```

## Generate Partitions and Final Dataset

The partitions used for training the models can be found here:
```
scp -r <user_name>@146.83.185.161:/home/db_storage/ztf_forced_photometry/partitions ./data/datasets/ZTF_ff 
```

Alternatively, you can utilize a previously created partition or assign a new one by setting the path in the path_save_k_fold variable within the ZTF_ff_processed_to_final.py script. If the specified partition does not exist, it will be created automatically. To generate both the partitions and the final dataset, run:

```
python data/ZTF_ff_processed_to_final.py
```

The quantiles are generated in this same script.

## Training

Execute the following commands, depending on the training components (this will generate a `results` file):

* Use the `configs/training.yaml`  to select which components of ATAT to train (only LC, LC + MD, LC + MD + Feat + MTA, etc...). It can be done specifying the hyperparameter `--experiment_type_general` as shown below.

```
# Train only with light curves (LC + MTA)
python training.py --experiment_type_general lc_mta --experiment_name_general experiment_0 --name_dataset_general ztf_ff --data_root_general data/datasets/ZTF_ff/final/LC_MD_FEAT_v3_windows_200_12

.
.
.

# Train with light curves and tabular information together (LC + MD + Feat + MTA)
python training.py --experiment_type_general lc_md_feat_mta --experiment_name_general experiment_0 --name_dataset_general ztf_ff --data_root_general data/datasets/ZTF_ff/final/LC_MD_FEAT_v3_windows_200_12
```

In the hyperparameter `--experiment_name_general` you can put any name you want. You could use `custom_parser.py` to configure the model hyperparameters.

## Evaluating Performance [ Estoy ordenando los Notebooks ]

After training the models, obtain the predictions and evaluate performance over various days (e.g., 16, 32, 64, 128, 256, 512, 1024, and 2048 days) using:

```
python inference_ztf.py ztf_ff
```

This generates a file called `predictions_times.pt` within each trained model's files. You could use the `notebooks/vis_results/ATAT_results_windows_v3.ipynb` to understand how obtain the metrics. It is just a notebook example that can variety depeding on you processing used (e.g. windows, logspace, linspace, etc...).

