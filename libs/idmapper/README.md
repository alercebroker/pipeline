# idmapper

`idmapper` is a Python library for converting catalog object IDs to master IDs and vice versa. It supports multiple catalogs including ZTF and LSST.

## Installation

To install the dependencies, use [Poetry](https://python-poetry.org/):

```sh
poetry install
```

## Usage

### Converting Catalog Object ID to Master ID

To convert a catalog object ID to a master ID, use the `catalog_oid_to_masterid` function:

```python
from idmapper.mapper import catalog_oid_to_masterid

catalog = "ZTF"
catalog_oid = "ZTF20abcdefg"
master_id = catalog_oid_to_masterid(catalog, catalog_oid)
print(master_id)
```

### Decoding Master ID

To decode a master ID back to its components, use the `decode_masterid` function:

```python
from idmapper.mapper import decode_masterid

master_id = 108086391180348693
catalog, catalog_oid = decode_masterid(master_id)
print(catalog, catalog_oid)
```

## Validation

To validate a ZTF object ID, use the `is_ztf_oid_valid` function:

```python
from idmapper.mapper import is_ztf_oid_valid

ztf_oid = "ZTF20abcdefg"
is_valid = is_ztf_oid_valid(ztf_oid)
print(is_valid)
```

## Running Tests

To run the tests, use `pytest`:

```sh
pytest
```
