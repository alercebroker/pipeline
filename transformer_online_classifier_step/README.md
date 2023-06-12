# Transformer Online Classifier

Use a transformer model for classify lightcurves in real time.

# Settings

The settings file requires:
- Database credentials to store the classifications.
- Kafka consumer and producer configuration.
- Kafka metrics configuration.
- The behaviour of the step depends on `CLASSIFIER` environment variable:
  - `header` (default) the step uses the Transformer of LC + Header model.
  - `features` the step uses the Transformer of LC + Header + Features model.
- The URL to binary files of models:
  - `MODEL_PATH`: URL to a specific model.
  - `HEADER_QUANTILES_PATH`: URL to quantiles transformer of header data.
  - `FEATURES_QUANTILES_PATH`: (only if `CLASSIFIER` is `features`) URL to quantiles transformer of features data.

