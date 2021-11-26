EXTRA_FIELDS = {
    "type": "map",
    "values": [
      "null",
      "int",
      "float",
      "string",
      "bytes"
    ],
    "default": {}
}

SCHEMA = {
  "type": "record",
  "doc": "Multi stream alert of any telescope/survey",
  "name": "alerce.alert",
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
      "type": ["null", "long"]
    },
    {
      "name": "candid",
      "type": "long"
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
      "name": "rb",
      "type": ["float", "null"]
    },
    {
      "name": "rbversion",
      "type": ["null", "string"]
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
      "name": "rfid",
      "type": ["null", "int"]
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
      "name": "extra_fields",
      "type": EXTRA_FIELDS,
    },
    {
      "name": "aid",
      "type": "long"
    },
  ]
}
