# An in-depth guide to the ALeRCE pipeline

Every step works as a "event driven microservice", which listens to a topic and must receive one or more messages in order to perform their operations. The pipeline steps are developed under the [APF Framework](https://github.com/alercebroker/APF). This framework gives some strong guidelines on the structure of the step, in addition of providing some hooks which facilitates the parsing of the information received by the step and the further deployment as well.  

## Pipeline Diagram (detailed)

![Image](https://user-images.githubusercontent.com/20263599/229163793-f0cefe89-6a2b-4dee-a111-20da2eec3461.png)

## Steps specification

### Sorting Hat

The Sorting Hat is a gateway step which assigns an ALeRCE ID (aid) to every incoming alert. This `aid` can be newly generated or an already existing one, depending on the result of a query performed on the object database. The query is based on whether the alert `oid` is already present in an object in the database, or if it lies within 1.5 arcsec of an object in the database.  
At the present time, this step must be deployed once per survey, with the same settings except of the topic it consumes and produces to. So you might find more than one instance of sorting hat deployed.

#### About the ALeRCE ID generation

The `aid` encodes the truncated position of the alert into a string. It always starts with AL and the last two digits of the current year.

#### Output Information

```python
{
    "type": "record",
    "doc": "Multi stream alert of any telescope/survey",
    "name": "alerce.alert",
    "fields": [
        {"name": "oid", "type": "string"},
        {"name": "tid", "type": "string"},
        {"name": "pid", "type": "long"},
        {"name": "candid", "type": ["long", "string"]},
        {"name": "mjd", "type": "double"},
        {"name": "fid", "type": "int"},
        {"name": "ra", "type": "double"},
        {"name": "dec", "type": "double"},
        {"name": "mag", "type": "float"},
        {"name": "e_mag", "type": "float"},
        {"name": "isdiffpos", "type": "int"},
        {"name": "e_ra", "type": "float"},
        {"name": "e_dec", "type": "float"},
        {
            "name": "extra_fields",
            "type": EXTRA_FIELDS,
        },
        {"name": "aid", "type": "string"},
        {"name": "stamps", "type": STAMPS},
    ],
}
```

Given the **STAMPS** type:

```python
{
    "type": "record",
    "name": "stamps",
    "fields": [
        {"name": "science", "type": ["null", "bytes"], "default": None},
        {"name": "template", "type": ["null", "bytes"], "default": None},
        {"name": "difference", "type": ["null", "bytes"], "default": None},
    ],
}
```

and the **EXTRA_FIELDS** type:

```python
{
    "type": "map",
    "values": ["null", "int", "float", "string", "bytes"],
    "default": {},
}
```

### Stamp Classifier Step

Applies a machine learning model to determine the classification of an alert based on its stamps. This should have an instance per survey, since the ML models are survey dependent.

#### Output Information

This step doesn't publish to any Kafka topic related to the pipeline.

### Previous Candidates Step

Parses the `extra_fields` of the detections to retrieve the binary field `prv_candidates` if the alert comes from ZTF. This will add to the main alert the previous detections and (possibly) non-detections. For other surveys, only the main alert is passed among the detections and the non-detections are empty. 

This step returns a list with the processed objects, including their `aid`, detections (main alert and previous detections), and non-detections.

#### Output Information

```python
{
  "type": "record",
  "doc": "Previous candidates schema with new alert and previous detections and non detections",
  "name": "prv_candidates",
  "fields": [
    {
      "name": "aid",
      "type": "string"
    },
    {
      "name": "detections",
      "type": {
        "type": "array",
        "items": {
          "type": "record",
          "name": "detection",
          "fields": [
            {
              "name": "oid",
              "type": "string"
            },
            {
              "name": "tid",
              "type": "string"
            },
            {
              "name": "pid",
              "type": "long"
            },
            {
              "name": "candid",
              "type": ["long", "string"]
            },
            {
              "name": "mjd",
              "type": "double"
            },
            {
              "name": "fid",
              "type": "int"
            },
            {
              "name": "ra",
              "type": "double"
            },
            {
              "name": "dec",
              "type": "double"
            },
            {
              "name": "mag",
              "type": "float"
            },
            {
              "name": "e_mag",
              "type": "float"
            },
            {
              "name": "isdiffpos",
              "type": "int"
            },
            {
              "name": "e_ra",
              "type": "float"
            },
            {
              "name": "e_dec",
              "type": "float"
            },
            {
              "name": "has_stamp",
              "type": "boolean"
            },
            {
              "name": "extra_fields",
              "type": {
                "default": {},
                "type": "map",
                "values": ["null", "int", "float", "string", "bytes"]
              }
            },
            {
              "name": "aid",
              "type": "string"
            }
          ]
        },
        "default": []
      }
    },
    {
      "name": "non_detections",
      "type": {
        "type": "array",
        "items": {
          "type": "record",
          "name": "non_detection",
          "fields": [
            {
              "name": "aid",
              "type": "string"
            },
            {
              "name": "tid",
              "type": "string"
            },
            {
              "name": "oid",
              "type": "string"
            },
            {
              "name": "mjd",
              "type": "double"
            },
            {
              "name": "fid",
              "type": "int"
            },
            {
              "name": "diffmaglim",
              "type": "int"
            }
          ]
        }
      },
      "default": []
    }
  ]
}
```

### Lightcurve Step 

This step retrieves from the database and then returns all stored detections and non-detections of an object, together with the new detections/non-detections coming from the previous step.

Repeated objects within the same batch of messages are merged.

#### Output Schema

The same output schema of the **Previous candidates step**.

### Correction Step

Process detections and applies correction to pass from difference magnitude to apparent magnitude.

It also calculates the object mean RA and declination based on the position of its detections.

#### Output Schema

```python
{
  "type": "record",
  "doc": "Previous candidates schema with new alert and previous detections and non detections",
  "name": "prv_candidates",
  "fields": [
    {
      "name": "aid",
      "type": "string"
    },
    {
      "name": "meanra",
      "type": "float"
    },
    {
      "name": "meandec",
      "type": "float"
    },
    {
      "name": "detections",
      "type": {
        "type": "array",
        "items": {
        "type": "record",
        "name": "alert",
        "fields": [
          {
            "name": "candid",
            "type": ["long", "string"]
          },
          {
            "name": "tid",
            "type": "string"
          },
          {
            "name": "aid",
            "type": "string"
          },
          {
            "name": "oid",
            "type": "string"
          },
          {
            "name": "mjd",
            "type": "double"
          },
          {
            "name": "fid",
            "type": "int"
          },
          {
            "name": "pid",
            "type": "long"
          },
          {
            "name": "ra",
            "type": "double"
          },
          {
            "name": "e_ra",
            "type": "float"
          },
          {
            "name": "dec",
            "type": "double"
          },
          {
            "name": "e_dec",
            "type": "float"
          },
          {
            "name": "mag",
            "type": "float"
          },
          {
            "name": "e_mag",
            "type": "float"
          },
          {
            "name": "mag_corr",
            "type": ["float", "null"]
          },
          {
            "name": "e_mag_corr",
            "type": ["float", "null"]
          },
          {
            "name": "e_mag_corr_ext",
            "type": ["float", "null"]
          },
          {
            "name": "isdiffpos",
            "type": "int"
          },
          {
            "name": "corrected",
            "type": "boolean"
          },
          {
            "name": "dubious",
            "type": "boolean"
          },
          {
            "name": "has_stamp",
            "type": "boolean"
          },
          {
            "name": "stellar",
            "type": "boolean"
          },
          {
            "name": "extra_fields",
            "type": {
              "default": {},
              "type": "map",
              "values": ["null", "int", "float", "string", "bytes", "boolean"]
            }
          }
        ]
      },
        "default": []
      }
    },
    {
      "name": "non_detections",
      "type": {
        "type": "array",
        "items": {
          "type": "record",
          "name": "non_detection",
          "fields": [
            {
              "name": "aid",
              "type": "string"
            },
            {
              "name": "tid",
              "type": "string"
            },
            {
              "name": "oid",
              "type": "string"
            },
            {
              "name": "mjd",
              "type": "double"
            },
            {
              "name": "fid",
              "type": "int"
            },
            {
              "name": "diffmaglim",
              "type": "int"
            }
          ]
        }
      },
      "default": []
    }
  ]
}
```

### Magstats Step

Generates/updates the object that is to be stored in the DB.

#### Output Schema

This step doesn't publish to any Kafka topic related to the pipeline.

### Xmatch Step

This step performs a crossmatch with ALLWISE catalog using CDS crossmatch service and sends the result to the xmatch topic of the Kafka server.

#### Output Schema
```python
{
    "doc": "Multi stream light curve with xmatch",
    "name": "alerce.light_curve_xmatched",
    "type": "record",
    "fields": [
        {"name": "aid", "type": "string"},
        {"name": "candid", "type": ["string", "long"]},
        {"name": "meanra", "type": "float"},
        {"name": "meandec", "type": "float"},
        {"name": "ndet", "type": "int"},
        {"name": "detections", "type": DETECTIONS},
        {"name": "non_detections", "type": NON_DETECTIONS},
        {"name": "xmatches", "type": [XMATCH, "null"]},
    ],
}
```

Given the `DETECTIONS` type:

```python
{
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "tid", "type": "string"},
            {"name": "candid", "type": ["string", "long"]},
            {"name": "mjd", "type": "double"},
            {"name": "fid", "type": "int"},
            {"name": "ra", "type": "double"},
            {"name": "e_ra", "type": "double"},
            {"name": "dec", "type": "double"},
            {"name": "e_dec", "type": "double"},
            {"name": "mag", "type": "float"},
            {"name": "e_mag", "type": "float"},
            {"name": "isdiffpos", "type": "int"},
            {"name": "rb", "type": ["float", "null"]},
            {"name": "rbversion", "type": ["string", "null"]},
            {
                "name": "extra_fields",
                "type": {
                    "type": "map",
                    "values": ["string", "int", "null", "float", "boolean", "double"],
                },
            },
        ],
    },
}
```

the `NON_DETECTIONS` type:


```python
{
    "type": "array",
    "items": {
        "name": "non_detections_record",
        "type": "record",
        "fields": [
            {"name": "tid", "type": "string"},
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "double"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "fid", "type": "int"},
        ],
    },
}
```

and the `XMATCH` type:


```python
{
    "type": "map",
    "values": {"type": "map", "values": ["string", "float", "null", "int"]},
}
```
### Features Step

Obtain the features of a given lightcurve, using FeatureExtractors.

#### Output Schema

```python
{
    "doc": "Features",
    "name": "features_document",
    "type": "record",
    "fields": [
        {"name": "oid", "type": "string"},
        {"name": "candid", "type": "long"},
        {"name": "aid", "type": "string"},
        {"name": "tid", "type": "string"},
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
                   {"name": "PPE", "type": ["float", "null"]},
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
        },
    ],
}
```

### LC Classifier

Perform a inference given a set of features. This might not get a classification if the object lacks of certain features.

#### Output Schema

This step doesn't publish to any Kafka topic.
