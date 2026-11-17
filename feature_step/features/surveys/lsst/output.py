"""lsst output messages, moved from features/utils/parsers.py."""
from typing import Dict, List  # type hints of the moved signature

from lc_classifier.features.core.base import AstroObject, query_ao_table  # object and metadata lookup

from features.surveys.common import flatten_features_for_output  # shared features dict for the output
from features.surveys.lsst.bands import LSST_BANDS  # lsst band table


def parse_output_lsst(  # signature spans several lines
    astro_objects: List[AstroObject],  # objects with computed features
    messages: List[Dict],  # messages as pre_execute left them, same order
    candids: Dict,  # oid -> measurement ids accumulated by execute
) -> list[dict]:  # returns one output message per object
    """
    Generate LSST-specific output messages from astro objects and their input messages.

    Each output contains:
      - oid
      - measurement_id (from candids[oid])
      - detections (as received in the input message)
      - features (flattened dict with band-suffixed names)

    Band suffix mapping follows LSST bands u,g,r,i,z,y -> _6,_1,_2,_3,_4,_5
    and combined colors like "g,r" -> _12 when present.
    """
    output_messages: list[dict] = []  # one message per object

    for message, astro_object in zip(messages, astro_objects):  # same order
        oid = message["oid"]  # object id of the message
        measurement_id = candids[oid]  # measurement ids accumulated by execute

        # Ensure AO metadata oid matches message oid
        oid_ao = query_ao_table(astro_object.metadata, "oid") #es esto necesario?
        assert oid_ao == oid  # message and object must match

        features_for_oid = flatten_features_for_output(astro_object, LSST_BANDS)  # name + band suffix -> value

        out_message = {  # output message
            "oid": oid,  # object id
            "measurement_id": measurement_id,  # measurement ids of the object
            "detections": message.get("detections", []), #nyj
            "features": features_for_oid, #features fueron calculados con microjy
            #features en que unidades quedaran.
        }  # end of the message
        output_messages.append(out_message)  # add it

    return output_messages  # one per object
