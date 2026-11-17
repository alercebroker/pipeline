"""ztf output messages, moved from features/utils/parsers.py."""
from typing import Dict, List  # type hints of the moved signature

from lc_classifier.features.core.base import AstroObject, query_ao_table  # object and metadata lookup

from features.surveys.common import flatten_features_for_output  # shared features dict for the output
from features.surveys.ztf.bands import ZTF_BANDS  # ztf band table


def parse_output(  # signature spans three lines
    astro_objects: List[AstroObject], messages: List[Dict], candids: Dict  # objects, prepared messages, oid -> candids
) -> list[dict]:  # returns one output message per object
    """
    Parse output of the step. It uses the input data to extend the schema to
    add the features of each object, identified by its oid.
    astro_objects and messages must be in the same order

    :param astro_objects:
    :param messages:
    :param candids:
    :return: a list of dictionaries, each input object with its data and the
        features calculated.
    """

    output_messages = []  # one message per object
    for message, astro_object in zip(messages, astro_objects):  # same order
        oid = message["oid"]  # object id of the message
        candid = candids[oid]  # candids accumulated by execute

        oid_ao = query_ao_table(astro_object.metadata, "oid")  # object id of the AstroObject
        assert oid_ao == oid  # message and object must match

        reference = astro_object.reference  # references of the object
        if reference is not None:  # object has references
            reference = astro_object.reference.reset_index(drop=True).to_dict("records")  # frame -> records

        features_for_oid = flatten_features_for_output(astro_object, ZTF_BANDS)  # name + band suffix -> value

        out_message = {  # output message
            "oid": oid,  # object id
            "candid": candid,  # candids of the object
            "detections": message["detections"], #photometria forzada
            "non_detections": message["non_detections"],  # non detections as received
            "xmatches": message.get("xmatches", None),  # allwise match or None
            "features": features_for_oid,  # computed features
            "reference": reference,  # removed by post_execute
        }  # end of the message
        output_messages.append(out_message)  # add it

    return output_messages  # one per object
