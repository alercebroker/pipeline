# Ingestion Step

## Description

The **ingestion_step** processes and inserts alert data from astronomical surveys (e.g., ZTF, LSST) into the ALeRCE multisurvey database. This step focuses on handling raw alert data, performing necessary transformations, and ensuring the data is ready for subsequent steps in the pipeline.

### Supported Surveys

- **ZTF**: Processes alerts from the Zwicky Transient Facility.
- **LSST**: Prepares for alerts from the Vera C. Rubin Observatory.

### Transformations

The following transformations are applied to the alert data:
- **`jd_to_mjd`**: Converts Julian Date to Modified Julian Date.
- **`fid_to_band`**: Maps filter IDs to band names.
- **`forcediffimflux_to_mag`**: Calculates magnitudes from flux values.
- **`isdiffpos_to_int`**: Converts `isdiffpos` to integer representation.

For a full list of transformations, refer to `transforms.py`.

## Local Installation

### Requirements

To install the repository without support for running the step (only including tools for ingestion), run:
```bash
pip install .
```

To include the step itself:
```bash
pip install .[apf]
```

### Development

Install development dependencies using [poetry](https://python-poetry.org/):
```bash
poetry install -E apf
```

Run tests using:
```bash
poetry run pytest
```

## Step Information

#### Next step:
- [Correction Multisurvey ZTF Step](https://github.com/alercebroker/pipeline/tree/main/correction_multisurvey_ztf_step)

## Run the Released Image

To pull the latest Docker image:
```bash
docker pull ghcr.io/alercebroker/ingestion_step:latest
```
