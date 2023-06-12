# Magnitude correction step

## Description

Process alerts and applies correction to pass from difference magnitude to apparent magnitude. This
includes the previous detections coming from the ZTF alerts.

### ZTF

The correction for ZTF is as follows:

$$m_\mathrm{corr} = -2.5 \log_{10}\left(10^{-0.4 m_\mathrm{ref}} + \mathrm{sgn} 10^{-0.4 m_\mathrm{diff}}\right)$$

For the error, the following is used:

$$\delta m_\mathrm{corr} = \frac{\sqrt{10^{-0.8 m_\mathrm{diff}} \delta m_\mathrm{diff}^2 - 10^{-0.8 m_\mathrm{ref}} \delta m_\mathrm{ref}^2}}{10^{-0.4 m_\mathrm{ref}} + \mathrm{sgn} 10^{-0.4 m_\mathrm{diff}}}$$

An additional error, for extended sources, is also calculated:
$$\delta m_\mathrm{corr} = \frac{10^{-0.4 m_\mathrm{diff}} \delta m_\mathrm{diff}}{10^{-0.4 m_\mathrm{ref}} + \mathrm{sgn} 10^{-0.4 m_\mathrm{diff}}}$$

### Others

Presently, there are no correction strategies for any other surveys.

Any other survey will get the correction fields added, but filled with `null` values.

## Local Installation

### Requirements

To install the repository without support for running the step (only including tools for correction) run:
```bash
pip install .
```

To include the step itself:
```bash
pip install .[apf]
```

#### Usage

```python
from correction import Corrector

detections = [{"aid": "AID1", "tid": "ZTF", ...: ..., "extra_fields": {"distnr": 1, ...: ...}}, ...]
corr = Corrector(detections)

corr.corrected  # Gets boolean pandas series on whether the detection can be corrected
corr.dubious  # Gets boolean pandas series on whether the correction/non-correction is dubious
corr.stellar  # Get boolean pandas series on whether the source is star-like
corr.corrected_magnitudes()  # Computes pandas dataframe with corrected magnitudes and errors
```

### Development

Including the development dependencies is only possible using [poetry](https://python-poetry.org/):
```bash
poetry install -E apf
```

Run tests using:
```bash
poetry run pytest
```

## Adding new strategies

New strategies (assumed to be survey based) can be added directly inside the module `core.strategy` as a new 
Python file. The name of the file must coincide with the survey name (lowercase), 
i.e., a file `atlas` will work on detections with `sid`s such as `ATLAS`, `AtLAs`, etc., but not `ATLAS-01`

Strategy modules are required to have 4 functions: 
* `is_corrected`: Returns boolean pandas data series showing if the detection can be corrected
* `is_dubious`: Returns boolean pandas data series showing whether the detection correction status is dubious
* `is_stellar`: Returns boolean pandas data series showing whether the detection is likely to be stellar
* `correct`: Returns pandas data frame with 3 columns (`mag_corr`, `e_mag_corr` and `e_mag_corr_ext`)

If detections with no survey strategy defined are part of the messages, these will be quietly filled with default 
values (`False` for the boolean fields and `NaN` for the corrected magnitudes).

**Important:** Remember to import the new module in `__init__.py` inside `core.strategy` or it won't be available.

## Step information

#### Previous step:
- [Light curve](https://github.com/alercebroker/lightcurve-step)

#### Next step:
- [Magnitude statistics](https://github.com/alercebroker/magstats_step)
- [Cross-match](https://github.com/alercebroker/xmatch_step)

## Environment variables

These are required only when running the scripts, as is the case for the Docker images.

### Consumer setup

- `CONSUMER_SERVER`: Kafka host with port, e.g., `localhost:9092`
- `CONSUMER_TOPICS`: Some topics. String separated by commas, e.g., `topic_one` or `topic_two,topic_three`
- `CONSUMER_GROUP_ID`: Name for consumer group, e.g., `correction`
- `CONSUME_TIMEOUT`: (optional) Timeout for consumer
- `CONSUME_MESSAGES`: (optional) Number of messages consumed in a batch

### Producer setup

- `PRODUCER_SERVER`: Kafka host with port, e.g., `localhost:9092`
- `PRODUCER_TOPIC`: Topic to write into for the next step

[//]: # (### SSL authentication)

[//]: # ()
[//]: # (When using SSL authentication for the whole cluster, the following must be provided)

[//]: # ()
[//]: # (- `KAFKA_USERNAME`: Username for the step authentication)

[//]: # (- `KAFKA_PASSWORD`: Password for the step authentication)

### Scribe producer setup

The [scribe](https://github.com/alercebroker/alerce-scribe) will write results in the database. 

- `SCRIBE_TOPIC`: Topic name, e.g., `topic_one`
- `SCRIBE_SERVER`: Kafka host with port, e.g., `localhost:9092`

### Metrics producer setup

- `METRICS_TOPIC`: (optional) Topic name, e.g., `topic_one`
- `METRICS_SERVER`: Kafka host with port, e.g., `localhost:9092`

## Run the released image

For each release, an image is uploaded to GitHub packages. To download:

```bash
docker pull ghcr.io/alercebroker/correction_step:latest
```
