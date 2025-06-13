# MagStats Step

## Development
Tests are yet to do and the project is not yet on a working state.
This project uses poetry as project manager.

## Description

Calculate new magstats for lightcurves processed by last steps.

This step performs:

- calculate new magstats
- insert magstats

#### Previous steps:

- Not determined

#### Next steps:

- None

## Database interactions


### Select:

- Query to get PS1 and Reference data.
- Query to get the previous magstats.

### Insert:

- New magstats.

## Previous conditions

No special conditions, only connection to kafka and database.


## Libraries used

- [APF](https://github.com/alercebroker/APF)
- [LC Correction](https://github.com/alercebroker/lc_correction)

## Stream

This step require a consumer.

### Input schema

Not determined yet

### Output schema

Not determined yet

