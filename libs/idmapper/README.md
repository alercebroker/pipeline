# idmapper

`idmapper` converts ZTF object ids (e.g. `ZTF20abcdefg`) to the int64 `oid` stored in the multisurvey database, and back.

An object in the database is identified by `(oid, sid)`, where `sid` is the survey id from the `sid_lut` table (`idmapper.mapper.SURVEY_IDS`). An `oid` is unique only within its survey: two surveys can store the same `oid`, so queries and joins that span surveys must match on both columns.

idmapper exists only for ZTF, whose ids are names. Surveys that publish integer ids (LSST) store them as they come and don't use it. A survey whose ids don't fit in a bigint, such as LS4 with its 128-bit UUIDs, needs its own oid scheme; idmapper doesn't handle it.

A ZTF oid is the encoded ZTF name with a prefix in its top 8 bits. The prefix is the ZTF sid + 1, so it is 1 for every stored ZTF oid.

## Installation

To install the dependencies, use [Poetry](https://python-poetry.org/):

```sh
poetry install
```

## Usage

### Converting a ZTF object id to an oid

Use the `catalog_oid_to_masterid` function:

```python
from idmapper.mapper import catalog_oid_to_masterid

catalog = "ZTF"
catalog_oid = "ZTF20abcdefg"
master_id = catalog_oid_to_masterid(catalog, catalog_oid)
print(master_id)  # 36028957668018384
```

### Decoding an oid

To convert a ZTF oid back to its ZTF object id, use the `decode_masterid` function:

```python
from idmapper.mapper import decode_masterid

master_id = 36028957668018384
catalog, catalog_oid = decode_masterid(master_id)
print(catalog, catalog_oid)  # ZTF ZTF20abcdefg
```

Both functions only accept ZTF and raise `ValueError` for anything else.

## Validation

`catalog_oid_to_masterid` validates the name by default and raises `ValueError` for a malformed one, which would otherwise encode to a wrong oid. Pass `validate=False` only for names that were already validated.

To validate a ZTF object ID, use the `is_ztf_oid_valid` function:

```python

from idmapper.ztf import is_ztf_oid_valid

ztf_oid = "ZTF20abcdefg"
is_valid = is_ztf_oid_valid(ztf_oid)
print(is_valid)
```

## Running Tests

To run the tests, use `pytest`:

```sh
pytest
```
