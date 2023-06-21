# LC Classifier

## Description
- Classification with Lightcurve Features
- There are 4 models that can be run in this step
  * ZTF [`Hierarchical Random Forest`](https://github.com/alercebroker/late_classifier/blob/4a37f6ee6a6ce6726fca4976b5206cffda8128b5/late_classifier/classifier/models.py#L117)
  * Toretto: The ELASTICC Hierarchical Random Forest that uses only lightcurve features
  * Barney: The ELASTICC Hierarchical Random Forest that uses header data
  * Balto: The ELASTICC Transformer that only uses Lightcurve metadata
  * Messi: The ELASTICC Transformer that uses Lightcurve metadata and features

### Setup for ZTF Random Forest
- Download model from [S3 bucket](https://assets.alerce.online/pipeline/hierarchical_rf_1.0.1/). 
- Version 1.0.1 of model from Light Curve Classification Paper.

### Setup for Toretto
- Download model from [S3 Bucket](https://assets.alerce.online/pipeline/elasticc/random_forest/2.0.1/)
- Current version is 2.0.1

The following step configuration will be needed:

``` python
PREDICTOR_CONFIG = {
    "CLASS": lc_classification.predictors.toretto.toretto_predictor.TorettoPredictor,
    "PARAMS": {"model_path": "https://assets.alerce.online/pipeline/elasticc/random_forest/2.0.1/"},
    "PARSER_CLASS": "lc_classification.predictors.toretto.toretto_parser.TorettoParser",
}
```

### Setup for Balto
TODO
### Setup for Messi
TODO
### Setup for Barney
TODO

#### Previous steps:
- [LC Features](https://github.com/alercebroker/feature_step)

## Database interactions

### Insert/Update (via Scribe)

Schema of the scribe command:

```python
command = {
    "collection": "object",
    "type": "update_probabilities",
    "criteria": {"_id": aid},
    "data": {
        "classifier_name": name,
        "classifier_version": version,
        "CLASS_NAME_1": probability1,
        "CLASS_NAME_2": probability2,
        ...
    },
    "options": {"upsert": True, "set_on_insert": False},
}
```

## Environment variables

- `STREAM`: Name of the stream to be consumed in lower caps. e.g: `ztf` or `elasticc`

    This will set the output schema of the step.
    This variable will be used for tests.

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `correction`
- `CONSUMER_CLASS`: Class of the consumer object. e.g: `apf.consumers.KafkaConsumer`
- `CONSUMER_KAFKA_USERNAME`: authentication for the consumer
- `CONSUMER_KAFKA_PASSWORD`: authentication for the consumer
### Producer setup

- `PRODUCER_TOPIC`: Name of output topic. e.g: `correction`
- `PRODUCER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `PRODUCER_CLASS`: Class of the producer object. e.g: `apf.producers.KafkaProducer`
- `PRODUCER_KAFKA_USERNAME`: authentication for the consumer
- `PRODUCER_KAFKA_PASSWORD`: authentication for the consumer

### Metrics setup

- `METRICS_HOST`: Kafka host for storing metrics
- `METRICS_TOPIC`: Name of the topic to store metrics
- `METRICS_KAFKA_USERNAME`: authentication for the consumer
- `METRICS_KAFKA_PASSWORD`: authentication for the consumer

### Scribe setup

- `SCRIBE_TOPIC`: Name of output topic. Now just uses `w_object`.
- `SCRIBE_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `SCRIBE_KAFKA_USERNAME`: authentication for the consumer
- `SCRIBE_KAFKA_PASSWORD`: authentication for the consumer

## Stream

### Input schema

- Output stream of [`Feature Step`](https://github.com/alercebroker/feature_step#output-schema).

### Output schema for ALeRCE

```python
{
    'doc': 'Late Classification',
    'name': 'probabilities_and_features',
    'type': 'record',
    'fields': [
        {'name': 'oid', 'type': 'string'},
        FEATURES_SCHEMA,
        {
            'name': 'lc_classification',
            'type': {
                'type': 'record',
                'name': 'late_record',
                'fields': [
                    {
                        'name': 'probabilities',
                        'type': {
                            'type': 'map',
                            'values': ['float'],
                        }
                    },
                    {
                        'name': 'class',
                        'type': 'string'
                    },
                    {
                        'name': 'hierarchical',
                        'type':
                        {
                            'name': 'root',
                            'type': 'map',
                            'values': [
                                {
                                    'type': 'map',
                                    'values': 'float'
                                },
                                {
                                    'type': 'map',
                                    'values': {
                                        'type': 'map',
                                        'values': 'float'
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
    ]
}
```

### Output schema for ELASTICC
TODO

#### FEATURES_SCHEMA for ZTF
```json
{
  "name": "features",
  "type": {
      "name": "features_record",
      "type": "record",
        "fields": [
         {"name": "Amplitude_1", "type": ["float", "null"]},
         {"name": "Amplitude_2", "type": ["float", "null"]},
         {"name": "AndersonDarling_1", "type": ["float", "null"]},
         {"name": "AndersonDarling_2", "type": ["float", "null"]},
         {"name": "Autocor_length_1", "type": ["double", "null"]},
         {"name": "Autocor_length_2", "type": ["double", "null"]},
         {"name": "Beyond1Std_1", "type": ["float", "null"]},
         {"name": "Beyond1Std_2", "type": ["float", "null"]},
         {"name": "Con_1", "type": ["double", "null"]},
         {"name": "Con_2", "type": ["double", "null"]},
         {"name": "Eta_e_1", "type": ["float", "null"]},
         {"name": "Eta_e_2", "type": ["float", "null"]},
         {"name": "ExcessVar_1", "type": ["double", "null"]},
         {"name": "ExcessVar_2", "type": ["double", "null"]},
         {"name": "GP_DRW_sigma_1", "type": ["double", "null"]},
         {"name": "GP_DRW_sigma_2", "type": ["double", "null"]},
         {"name": "GP_DRW_tau_1", "type": ["float", "null"]},
         {"name": "GP_DRW_tau_2", "type": ["float", "null"]},
         {"name": "Gskew_1", "type": ["float", "null"]},
         {"name": "Gskew_2", "type": ["float", "null"]},
         {"name": "Harmonics_mag_1_1", "type": ["float", "null"]},
         {"name": "Harmonics_mag_1_2", "type": ["float", "null"]},
         {"name": "Harmonics_mag_2_1", "type": ["float", "null"]},
         {"name": "Harmonics_mag_2_2", "type": ["float", "null"]},
         {"name": "Harmonics_mag_3_1", "type": ["float", "null"]},
         {"name": "Harmonics_mag_3_2", "type": ["float", "null"]},
         {"name": "Harmonics_mag_4_1", "type": ["float", "null"]},
         {"name": "Harmonics_mag_4_2", "type": ["float", "null"]},
         {"name": "Harmonics_mag_5_1", "type": ["float", "null"]},
         {"name": "Harmonics_mag_5_2", "type": ["float", "null"]},
         {"name": "Harmonics_mag_6_1", "type": ["float", "null"]},
         {"name": "Harmonics_mag_6_2", "type": ["float", "null"]},
         {"name": "Harmonics_mag_7_1", "type": ["float", "null"]},
         {"name": "Harmonics_mag_7_2", "type": ["float", "null"]},
         {"name": "Harmonics_mse_1", "type": ["double", "null"]},
         {"name": "Harmonics_mse_2", "type": ["double", "null"]},
         {"name": "Harmonics_phase_2_1", "type": ["float", "null"]},
         {"name": "Harmonics_phase_2_2", "type": ["float", "null"]},
         {"name": "Harmonics_phase_3_1", "type": ["float", "null"]},
         {"name": "Harmonics_phase_3_2", "type": ["float", "null"]},
         {"name": "Harmonics_phase_4_1", "type": ["float", "null"]},
         {"name": "Harmonics_phase_4_2", "type": ["float", "null"]},
         {"name": "Harmonics_phase_5_1", "type": ["float", "null"]},
         {"name": "Harmonics_phase_5_2", "type": ["float", "null"]},
         {"name": "Harmonics_phase_6_1", "type": ["float", "null"]},
         {"name": "Harmonics_phase_6_2", "type": ["float", "null"]},
         {"name": "Harmonics_phase_7_1", "type": ["float", "null"]},
         {"name": "Harmonics_phase_7_2", "type": ["float", "null"]},
         {"name": "IAR_phi_1", "type": ["double", "null"]},
         {"name": "IAR_phi_2", "type": ["float", "null"]},
         {"name": "LinearTrend_1", "type": ["float", "null"]},
         {"name": "LinearTrend_2", "type": ["double", "null"]},
         {"name": "MHPS_PN_flag_1", "type": ["double", "null"]},
         {"name": "MHPS_PN_flag_2", "type": ["double", "null"]},
         {"name": "MHPS_high_1", "type": ["float", "null"]},
         {"name": "MHPS_high_2", "type": ["double", "null"]},
         {"name": "MHPS_low_1", "type": ["float", "null"]},
         {"name": "MHPS_low_2", "type": ["float", "null"]},
         {"name": "MHPS_non_zero_1", "type": ["double", "null"]},
         {"name": "MHPS_non_zero_2", "type": ["double", "null"]},
         {"name": "MHPS_ratio_1", "type": ["float", "null"]},
         {"name": "MHPS_ratio_2", "type": ["float", "null"]},
         {"name": "MaxSlope_1", "type": ["float", "null"]},
         {"name": "MaxSlope_2", "type": ["float", "null"]},
         {"name": "Mean_1", "type": ["float", "null"]},
         {"name": "Mean_2", "type": ["float", "null"]},
         {"name": "Meanvariance_1", "type": ["float", "null"]},
         {"name": "Meanvariance_2", "type": ["float", "null"]},
         {"name": "MedianAbsDev_1", "type": ["float", "null"]},
         {"name": "MedianAbsDev_2", "type": ["float", "null"]},
         {"name": "MedianBRP_1", "type": ["float", "null"]},
         {"name": "MedianBRP_2", "type": ["float", "null"]},
         {"name": "Multiband_period", "type": ["float", "null"]},
         {"name": "PairSlopeTrend_1", "type": ["float", "null"]},
         {"name": "PairSlopeTrend_2", "type": ["float", "null"]},
         {"name": "PercentAmplitude_1", "type": ["float", "null"]},
         {"name": "PercentAmplitude_2", "type": ["float", "null"]},
         {"name": "Period_band_1", "type": ["float", "null"]},
         {"name": "Period_band_2", "type": ["float", "null"]},
         {"name": "delta_period_1", "type": ["float", "null"]},
         {"name": "delta_period_2", "type": ["float", "null"]},
         {"name": "Period_fit", "type": ["float", "null"]},
         {"name": "Power_rate_1/2", "type": ["float", "null"]},
         {"name": "Power_rate_1/3", "type": ["float", "null"]},
         {"name": "Power_rate_1/4", "type": ["float", "null"]},
         {"name": "Power_rate_2", "type": ["float", "null"]},
         {"name": "Power_rate_3", "type": ["float", "null"]},
         {"name": "Power_rate_4", "type": ["float", "null"]},
         {"name": "Psi_CS_1", "type": ["float", "null"]},
         {"name": "Psi_CS_2", "type": ["float", "null"]},
         {"name": "Psi_eta_1", "type": ["float", "null"]},
         {"name": "Psi_eta_2", "type": ["float", "null"]},
         {"name": "Pvar_1", "type": ["float", "null"]},
         {"name": "Pvar_2", "type": ["float", "null"]},
         {"name": "Q31_1", "type": ["float", "null"]},
         {"name": "Q31_2", "type": ["float", "null"]},
         {"name": "Rcs_1", "type": ["float", "null"]},
         {"name": "Rcs_2", "type": ["float", "null"]},
         {"name": "SF_ML_amplitude_1", "type": ["float", "null"]},
         {"name": "SF_ML_amplitude_2", "type": ["float", "null"]},
         {"name": "SF_ML_gamma_1", "type": ["float", "null"]},
         {"name": "SF_ML_gamma_2", "type": ["float", "null"]},
         {"name": "SPM_A_1", "type": ["float", "null"]},
         {"name": "SPM_A_2", "type": ["float", "null"]},
         {"name": "SPM_beta_1", "type": ["float", "null"]},
         {"name": "SPM_beta_2", "type": ["float", "null"]},
         {"name": "SPM_chi_1", "type": ["float", "null"]},
         {"name": "SPM_chi_2", "type": ["float", "null"]},
         {"name": "SPM_gamma_1", "type": ["float", "null"]},
         {"name": "SPM_gamma_2", "type": ["float", "null"]},
         {"name": "SPM_t0_1", "type": ["float", "null"]},
         {"name": "SPM_t0_2", "type": ["float", "null"]},
         {"name": "SPM_tau_fall_1", "type": ["float", "null"]},
         {"name": "SPM_tau_fall_2", "type": ["float", "null"]},
         {"name": "SPM_tau_rise_1", "type": ["float", "null"]},
         {"name": "SPM_tau_rise_2", "type": ["float", "null"]},
         {"name": "Skew_1", "type": ["float", "null"]},
         {"name": "Skew_2", "type": ["float", "null"]},
         {"name": "SmallKurtosis_1", "type": ["float", "null"]},
         {"name": "SmallKurtosis_2", "type": ["float", "null"]},
         {"name": "Std_1", "type": ["float", "null"]},
         {"name": "Std_2", "type": ["float", "null"]},
         {"name": "StetsonK_1", "type": ["float", "null"]},
         {"name": "StetsonK_2", "type": ["float", "null"]},
         {"name": "W1-W2", "type": ["double", "null"]},
         {"name": "W2-W3", "type": ["double", "null"]},
         {"name": "delta_mag_fid_1", "type": ["float", "null"]},
         {"name": "delta_mag_fid_2", "type": ["float", "null"]},
         {"name": "delta_mjd_fid_1", "type": ["float", "null"]},
         {"name": "delta_mjd_fid_2", "type": ["float", "null"]},
         {"name": "dmag_first_det_fid_1", "type": ["double", "null"]},
         {"name": "dmag_first_det_fid_2", "type": ["double", "null"]},
         {"name": "dmag_non_det_fid_1", "type": ["double", "null"]},
         {"name": "dmag_non_det_fid_2", "type": ["double", "null"]},
         {"name": "first_mag_1", "type": ["float", "null"]},
         {"name": "first_mag_2", "type": ["float", "null"]},
         {"name": "g-W2", "type": ["double", "null"]},
         {"name": "g-W3", "type": ["double", "null"]},
         {"name": "g-r_max", "type": ["float", "null"]},
         {"name": "g-r_max_corr", "type": ["float", "null"]},
         {"name": "g-r_mean", "type": ["float", "null"]},
         {"name": "g-r_mean_corr", "type": ["float", "null"]},
         {"name": "gal_b", "type": ["float", "null"]},
         {"name": "gal_l", "type": ["float", "null"]},
         {"name": "iqr_1", "type": ["float", "null"]},
         {"name": "iqr_2", "type": ["float", "null"]},
          {
              "name": "last_diffmaglim_before_fid_1",
              "type": ["double", "null"],
          },
          {
              "name": "last_diffmaglim_before_fid_2",
              "type": ["double", "null"],
          },
          {"name": "last_mjd_before_fid_1", "type": ["double", "null"]},
          {"name": "last_mjd_before_fid_2", "type": ["double", "null"]},
          {
              "name": "max_diffmaglim_after_fid_1",
              "type": ["double", "null"],
          },
          {
              "name": "max_diffmaglim_after_fid_2",
              "type": ["double", "null"],
          },
          {
              "name": "max_diffmaglim_before_fid_1",
              "type": ["double", "null"],
          },
          {
              "name": "max_diffmaglim_before_fid_2",
              "type": ["double", "null"],
          },
          {"name": "mean_mag_1", "type": ["float","null"]},
          {"name": "mean_mag_2", "type": ["float","null"]},
          {
              "name": "median_diffmaglim_after_fid_1",
              "type": ["double", "null"],
          },
          {
              "name": "median_diffmaglim_after_fid_2",
              "type": ["double", "null"],
          },
          {
              "name": "median_diffmaglim_before_fid_1",
              "type": ["double", "null"],
          },
          {
              "name": "median_diffmaglim_before_fid_2",
              "type": ["double", "null"],
          },
          {"name": "min_mag_1", "type": ["float", "null"]},
          {"name": "min_mag_2", "type": ["float", "null"]},
          {"name": "n_det_1", "type": ["double", "null"]},
          {"name": "n_det_2", "type": ["double", "null"]},
          {"name": "n_neg_1", "type": ["double", "null"]},
          {"name": "n_neg_2", "type": ["double", "null"]},
          {"name": "n_non_det_after_fid_1", "type": ["double", "null"]},
          {"name": "n_non_det_after_fid_2", "type": ["double", "null"]},
          {"name": "n_non_det_before_fid_1", "type": ["double", "null"]},
          {"name": "n_non_det_before_fid_2", "type": ["double", "null"]},
          {"name": "n_pos_1", "type": ["double", "null"]},
          {"name": "n_pos_2", "type": ["double", "null"]},
          {"name": "positive_fraction_1", "type": ["double", "null"]},
          {"name": "positive_fraction_2", "type": ["double", "null"]},
          {"name": "r-W2", "type": ["double", "null"]},
          {"name": "r-W3", "type": ["double", "null"]},
          {"name": "rb", "type": ["float", "null"]},
          {"name": "sgscore1", "type": ["float", "null"]}
      ],
  },
}
```


## Build docker image

For use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t late_classification_step:latest .
```

## Run step

### Run container of docker

You can use a `docker run` command, you must set all environment variables.

```bash
docker run --name my_step -e DB_NAME=myhost -e [... all env ...] -d late_classification_step:latest
```

### Run docker-compose

Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/feature_step/blob/master/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

**Note:** Use `docker-compose down` to stop all containers.

### Run the released image

For each release an image is uploaded to ghcr.io that you can use instead of building your own. To do that replace `docker-compose.yml` or the `docker run` command with this image:

```bash
docker pull ghcr.io/alercebroker/image_name:latest
```

## Local Installation

### Requirements

To install the required packages run

```bash
pip install -r requirements_ztf.txt
```

or

```bash
pip install -r requirements_elasticc.txt
```
### Poetry

#### Set-up poetry:
- Install poetry: `pip install poetry`
- If you want to set create `.venv` environment in the project folder: `poetry config virtualenvs.in-project true`
- Set github configuration, use poetry config or env variables.
  - `poetry config http-basic.git <username> <password>`
  - `export POETRY_HTTP_BASIC_GIT_USERNAME=<username>`
  - `export POETRY_HTTP_BASIC_GIT_PASSWORD=<gh_token>`
- Install desired environment: 
  - `poetry install --with ztf`
  - `poetry install --with toretto`
  - `poetry install --with messi`
  - `poetry install --with balto`
  - `poetry install --with elasticc`
- Add a new dependency 
  - `poetry add -G <group> PACKAGE`

#### Run command with poetry environment
- Run: `poetry run <command>`
### Tests
To run tests install 

```bash
pip install pytest pytest-docker
```

Then run the command for some test
    
```bash
python -m pytest tests/unit
```

or

```bash
python -m pytest tests/integration
```

**NOTE:** Remember to set the STREAM env variable
- `STREAM`: Name of the stream to be consumed in lower caps. e.g: `ztf` or `elasticc`