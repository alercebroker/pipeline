XMATCH = {
	'type' : 'map',
    'values' : {
        'type' : 'map',
        'values' : [ "string", "float", "null"]
    }
}

METADATA = {
	'type' : 'map',
    'values' : {
        'type' : 'map',
        'values' : [ "string", "float", "null"]
    }
}

SCHEMA = {
    'doc': 'Light curve',
    'name': 'light_curve',
    'type': 'record',
    'fields': [
        {'name': 'oid', 'type': 'string', 'default': None},
        {'name': 'candid', 'type': 'string'},
        {'name': 'fid', 'type': 'int'},
        {'name': 'detections', 'type': {
            'type': 'array',
            'items': {
                'type': 'map',
                'values': ['float', 'int', 'string', 'null', 'boolean']
            }
        }},
        {'name': 'non_detections', 'type': {
            'type': 'array',
            'items': {
                'type': 'map',
                'values': ['float', 'int', 'string', 'null']
            }
        }},
        {'name': 'xmatches', 'type': [XMATCH, "null"], "default": "null"},
        {'name': 'metadata', 'type': [METADATA, "null"], "default": "null"}
    ],
}
