


from db_plugins.db.sql.models_new import ZtfDetection, ZtfForcedPhotometry, ForcedPhotometry, NonDetection, Detection
import math
# We assign tid and sid to 0, subject to changes in the future

GENERIC_FIELDS = [
        "tid",
        "sid",
        "oid",
        "pid",
        "mjd",
        "fid",
        "ra",
        "dec",
        'measurement_id',
        "isdiffpos",
        "parent_candid",
        "has_stamp",
        'magpsf',
        'sigmapsf',
        'mag',
        'e_mag',
    ]

CHANGE_VALUES = [
        "tid",
        "sid",
]

GENERIC_FIELDS_FP = [
        "tid",
        "sid",
        "oid",
        "pid",
        "mjd",
        "fid",
        "ra",
        "dec",
        "isdiffpos",
        "parent_candid",
        "has_stamp",
    ]

CHANGE_NAMES = { # outside extrafields
            "magpsf": "mag",
            "sigmapsf": "e_mag",
        }

CHANGE_NAMES_2 = { # inside extrafields
            "sigmapsf_corr": "e_mag_corr",
            "sigmapsf_corr_ext": "e_mag_corr_ext",
            "magpsf_corr": "mag_corr",
    }
ERRORS = {
    1: 0.065,
    2: 0.085,
    3: 0.01,
}
    
def _e_ra(dec, fid):
    try:
        return ERRORS[fid] / abs(math.cos(math.radians(dec)))
    except ZeroDivisionError:
        return float("nan")

def parse_sql_detection(ztf_models: list, models: list, *, oids) -> list:
    """
    Here we join the ztf_detections and detections in one dict for every oid. Also we add the field extra_fields to the final result
    and we change some names helped by the CHANGE_NAMES dict in the top of the code. 

    """
    parsed_ztf_dets = []
    extra_fields_list = []

    # Here we take all the info from the DDBB and split it into dictionaries. Also we save some variables called extra_fields where this 
    # variables are NOT in the GENERIC_FIELDS dict.
    for det in ztf_models:
        det: dict = det[0].__dict__
        extra_fields = {}
        parsed_det = {}
        for field, value in det.items():
            if field.startswith("_"):
                continue
            elif not field in GENERIC_FIELDS:
                extra_fields[field] = value
            else:
                if field in CHANGE_VALUES:
                    parsed_det[field] = 0
                else:
                    parsed_det[field] = value
        parsed_ztf_dets.append(parsed_det)
        extra_fields_list.append(extra_fields)

    parsed_dets = []

    # Here we take all the info from the DDBB and split it into dictionaries too.
    for d in models:
        d: dict = d[0].__dict__
        parsed_d = {}
        for field, value in d.items():
            if field.startswith("_"):
                continue
            else:
                if field in CHANGE_VALUES:
                    parsed_d[field] = 0
                else:
                    parsed_d[field] = value
        parsed_dets.append(parsed_d)

    parsed_ztf_list = []

    # Here we form the parsed_ztf object for every dict in parsed_ztf_dets. 
    for det in parsed_ztf_dets:
        parsed_ztf = ZtfDetection(
            **det,
        )
        parsed_ztf_list.append(parsed_ztf)

    # Here we join detections and ztf_detections in one. Also we hardcode forced and new as False because this is from the DDBB
    # and is not a new object. On the other land as we are in detections, this is not forced so forced is false. Also we join the extra_fields field 
    # and change some names in this field.
    for detections in parsed_dets:
        for key, value in detections.items():
            if not key in parsed_ztf_dets[parsed_dets.index(detections)].keys():  
                setattr(parsed_ztf_list[parsed_dets.index(detections)], key, value)
    
        setattr(parsed_ztf_list[parsed_dets.index(detections)], 'forced', False)
        setattr(parsed_ztf_list[parsed_dets.index(detections)], 'new', False)
        setattr(parsed_ztf_list[parsed_dets.index(detections)], 'extra_fields', extra_fields_list[parsed_dets.index(detections)])
        for name in CHANGE_NAMES.keys():
            parsed_ztf_list[parsed_dets.index(detections)].__dict__[CHANGE_NAMES[name]] = parsed_ztf_list[parsed_dets.index(detections)].__dict__[name]
            del parsed_ztf_list[parsed_dets.index(detections)].__dict__[name]

        for name in CHANGE_NAMES_2.keys():
            parsed_ztf_list[parsed_dets.index(detections)].__dict__['extra_fields'][CHANGE_NAMES_2[name]] = parsed_ztf_list[parsed_dets.index(detections)].__dict__['extra_fields'][name]
            del parsed_ztf_list[parsed_dets.index(detections)].__dict__['extra_fields'][name]

    dict_parsed = list(map(vars,parsed_ztf_list))

    # Here we calculate e_ra and e_dec and join into every final dict. These variables are used to calculate meanra and meandec in correction.
    for d in dict_parsed:
        e_ra  = _e_ra(d['dec'], d['band'])
        e_dec = ERRORS[d['band']]
        
        d['e_ra'] = e_ra
        d['e_dec'] = e_dec

        d['sid'] = 0
        d['tid'] = 0
    
    for d in dict_parsed:
        del d['_sa_instance_state']

    return dict_parsed


def parse_sql_non_detection(ztf_models: list, *, oids) -> list:
    non_dets = []
    
    for d in ztf_models:
        parsed_non_det = {}
        for field, value in d[0].__dict__.items():
            if field.startswith("_"):
                continue
            else:
                if field in CHANGE_VALUES:
                    parsed_non_det[field] = 0
                else:
                    parsed_non_det[field] = value
        non_dets.append(parsed_non_det)

    non_dets_parsed = []
    for non_det in non_dets:
        non_detection = NonDetection(
            **non_det
        )
        non_dets_parsed.append(non_detection)

    dict_parsed = list(map(vars,non_dets_parsed))
    for d in dict_parsed:
        del d['_sa_instance_state']
        d['sid'] = 0
        d['tid'] = 0
        
    

    return dict_parsed

def parse_sql_forced_photometry(ztf_models: list, models: list, *, oids) -> list:
    """
    Here we join the ztf_fp and fp in one dict for every oid. Also we add the field extra_fields to the final result.
    """
    parsed_ztf_dets = []
    extra_fields_list = []
    # Here we take the info from the DDBB and make the extra_fields list of dicts.
    for d in ztf_models:
        parsed_fp_d = {}
        extra_fields = {}
        for key, value in d[0].__dict__.items():
            if key.startswith("_"):
                continue
            elif not key in GENERIC_FIELDS:
                extra_fields[key] = value
            else:
                if key in CHANGE_VALUES:
                    parsed_fp_d[key] = 0
                else:
                    parsed_fp_d[key] = value
        parsed_ztf_dets.append(parsed_fp_d)
        extra_fields_list.append(extra_fields)
    parsed_dets = []

    # Here we take the info from the DDBB and make the extra_fields list of dicts.
    for d in models:
        d: dict = d[0].__dict__
        parsed_d = {}
        for field, value in d.items():
            if field.startswith("_"):
                continue
            else:
                if field in CHANGE_VALUES:
                    parsed_d[field] = 0
                else:
                    parsed_d[field] = value
        parsed_dets.append(parsed_d)

    parsed_ztf_list = []

    for forced in parsed_ztf_dets:
        parsed_ztf = ZtfForcedPhotometry(
                        **forced
                    ),
        parsed_ztf_list.append(parsed_ztf[0])

    # Here we join ztf_fp and fp and put forced to true and new to false.
    for detections in parsed_dets:
        for key, value in detections.items():
            if not key in parsed_ztf_dets[parsed_dets.index(detections)].keys():  
                setattr(parsed_ztf_list[parsed_dets.index(detections)], key, value)
    
        setattr(parsed_ztf_list[parsed_dets.index(detections)], 'forced', True)
        setattr(parsed_ztf_list[parsed_dets.index(detections)], 'new', False)
        setattr(parsed_ztf_list[parsed_dets.index(detections)], 'extra_fields', extra_fields_list[parsed_dets.index(detections)])

    dict_parsed = list(map(vars,parsed_ztf_list))

    for d in dict_parsed:
        del d['_sa_instance_state']
        d['sid'] = 0
        d['tid'] = 0
        del d['pid'] # Temporal fix to avoid error (Not all pids have value in the new DDBB)
        d['pid'] = 0
    return dict_parsed