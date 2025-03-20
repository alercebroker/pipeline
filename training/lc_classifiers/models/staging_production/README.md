# Testing ALeRCE Classifier Models in Staging

First, you should have completed the Data Acquisition step to obtain the required data.

## Data Acquisition 

To perform inference with ATAT, you need to obtain the preprocessed data and partitions from the [`data_acquisition folder`](../../../data_acquisition). Detailed explanations are provided in the [`README.MD`](../../../data_acquisition/README.MD). In general, the following data is required to prepare the model's input:

1. Pickle files containing dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py) with the following keys: `'detections'`, `'non_detections'`, and `'forced_photometry'`.

2. Pickle files containing dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py) with the `'features'`.

3. Parquet files containing partitions.

## Inference

You should configure the variables for the model you want to use, which is uploaded to S3, in the `.env` file.

```
pip install python-dotenv
```

Then, you can use the `inference.py` script. Ensure the file path is correctly set at the end of the script. After running it, you will obtain a Parquet file with predictions in the results folder, which you can visualize using the provided notebooks.

## Notes

For additional support, refer to the documentation or contact the ALeRCE team.


