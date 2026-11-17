"""lsst scribe commands, moved from features/utils/parsers.py."""
from typing import List  # type hints of the moved signature

import pandas as pd  # return type of prepare_ao_features_for_db_lsst
from lc_classifier.features.core.base import AstroObject, query_ao_table  # object and metadata lookup

from features.surveys.common import prepare_features_for_db  # shared feature rows for the db
from features.surveys.lsst.bands import LSST_BANDS  # lsst band table


def fid_mapper_for_db_lsst(band: str) -> int:
    """
    Map LSST band identifiers to DB fid codes using explicit mapping.

    Singles: u,g,r,i,z,y -> 6,1,2,3,4,5 #esto cambiar
    Valid combinations: u,g g,r r,i i,z z,y -> 61,12,23,34,45
    """
    return LSST_BANDS.fid_for_db(band)  # None guard, str() and strip() live in the lsst band table


def prepare_ao_features_for_db_lsst(astro_object: AstroObject, feature_name_lut) -> pd.DataFrame:
    """Return the lsst feature rows for the scribe upsert, without the name column."""
    return prepare_features_for_db(astro_object, feature_name_lut, LSST_BANDS, keep_name=False)  # shared body, lsst bands


def parse_scribe_payload_lsst(  # signature spans three lines
    astro_objects: List[AstroObject], features_version, features_group, feature_name_lut  # objects, version, group, LUT
):  # returns the commands dict
    """Create scribe commands for LSST without color updates.

    Builds only update_features commands and returns them under
    the 'upserting_features' key. No update_object commands are
    created or returned.
    """
    upsert_features_commands_list = []  # one features command per object

    for astro_object in astro_objects:  # every computed object
        ao_features = prepare_features_for_db(astro_object, feature_name_lut, LSST_BANDS, keep_name=False)  # lsst rows, no name
        oid = query_ao_table(astro_object.metadata, "oid")  # object id
        last_mjd = query_ao_table(astro_object.metadata, "last_mjd")  # newest epoch


        features_list = ao_features.to_dict("records")  # feature rows as dicts
        sid = astro_object.detections["sid"].values[0]  # survey id of the object
        # New envelope format required downstream
        upsert_features_command = {  # features upsert command
            "step": "features", #cual tiene que ser el nombre aqui?
            "survey": "lsst",  # survey of the command
            "payload": {  # rows to upsert
                "oid": int(oid),  # object id
                "features_version": features_version,  # version id
                "sid": sid,  # survey id
                "mjd": last_mjd,  # newest epoch
                "features": features_list,  # feature rows
            },  # end of the payload
        }  # end of the command
        upsert_features_commands_list.append(upsert_features_command)  # add it

    return {"upserting_features": upsert_features_commands_list}  # lsst sends no update_object
