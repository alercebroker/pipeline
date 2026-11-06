"""ztf scribe commands, moved from features/utils/parsers.py."""
from typing import List  # type hints of the moved signature

from lc_classifier.features.core.base import AstroObject, query_ao_table  # object and metadata lookup

from features.surveys.common import prepare_features_for_db  # shared feature rows for the db
from features.surveys.ztf.bands import ZTF_BANDS  # ztf band table


def fid_mapper_for_db(band: str):
    """
    Parses the number used to reference the fid in the ztf alerts
    to the string value corresponding
    """
    return ZTF_BANDS.fid_for_db(band)  # g 1, r 2, g,r 12, anything else 0


def parse_scribe_payload(  # signature spans three lines
    astro_objects: List[AstroObject], features_version, features_group,feature_name_lut  # objects, version, group, LUT
):  # returns the commands dict
    """Create the json with the messages for the scribe producer from the
    features dataframe. It adds the fid and correct the name.

    :param astro_objects: a list of AstroObjects with computed features inside.
    :param features_version: a string with the features version used
    :return: a list of json with Alerce Scribe commands
    """

    # features = features.replace({np.nan: None, np.inf: None, -np.inf: None})
    upsert_features_commands_list = []  # one features command per object
    update_object_command_list = []  # one update-object command per object

    for astro_object in astro_objects:  # every computed object
        # for upserting features
        ao_features = prepare_features_for_db(astro_object, feature_name_lut, ZTF_BANDS, keep_name=True)  # ztf rows, name kept
        oid = query_ao_table(astro_object.metadata, "oid")  # object id
        last_mjd = query_ao_table(astro_object.metadata, "last_mjd")  # newest epoch

        features_list = ao_features.to_dict("records")  # rows with names, for the colors
        # for updating the object
        def get_color_from_features(name, features_list):  # color value of `name` in band 12
            color = list(  # rows matching the name in the g,r band
                filter(lambda x: x["name"] == name and x["band"] == 12, features_list)  # name and band 12
            )  # end of the list
            color = color[0]["value"] if len(color) == 1 else None  # exactly one match, else None
            return color  # color value or None

        update_object_command = {  # update-ztf-object-features command
            "survey": "ztf",  # survey of the command
            "step": "update-ztf-object-features",  # scribe step
            "payload": {  # values to write on the object
                "oid": int(oid),  # object id
                "mjd": last_mjd,  # newest epoch
                "g_r_max": get_color_from_features("g-r_max", features_list),  # g-r max color
                "g_r_mean": get_color_from_features("g-r_mean", features_list),  # g-r mean color
                "g_r_max_corr": get_color_from_features("g-r_max_corr", features_list),  # corrected g-r max
                "g_r_mean_corr": get_color_from_features(  # corrected g-r mean
                    "g-r_mean_corr", features_list  # name and rows
                ),  # end of the call
            },  # end of the payload
            "criteria": {"oid": int(oid)},  # row to update
            "options": {},  # no options
        }  # end of the command
        update_object_command_list.append(update_object_command)  # add it


        ao_features.drop(columns=["name"], inplace=True)  # the upsert does not carry names
        features_list = ao_features.to_dict("records")  # rows without names

        sid = astro_object.detections["sid"].values[0]  # survey id of the object
        # New envelope format required downstream
        upsert_features_command = {  # features upsert command
            "step": "features", #cual tiene que ser el nombre aqui?
            "survey": "ztf",  # survey of the command
            "payload": {  # rows to upsert
                "oid": int(oid),  # object id
                "features_version": features_version,  # version id
                "sid": sid,  # survey id
                "features": features_list,  # feature rows
                "mjd": last_mjd,  # newest epoch
            },  # end of the payload
        }  # end of the command
        upsert_features_commands_list.append(upsert_features_command) #esto tengo que hacerlo

    return {  # both command lists
        "update_object": update_object_command_list,  # object color updates
        "upserting_features": upsert_features_commands_list,  # feature upserts
    }  # end of the commands dict
