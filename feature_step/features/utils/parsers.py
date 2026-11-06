import logging
import numpy as np
import pandas as pd
from lc_classifier.features.core.base import AstroObject, query_ao_table
from lc_classifier.utils import mag2flux, mag_err_2_flux_err
from typing import List, Dict, Optional
import os
import json
from features.surveys.common import last_mjd_of, prepare_features_for_db, wise_magnitudes  # shared survey helpers
from features.surveys.lsst import LSSTSurvey  # lsst band table
from features.surveys.ztf import ZTFSurvey  # ztf band table
from features.surveys.common import DETECTION_KEYS_MAP  # moved to common, re-exported for lsst and importers
from features.surveys.ztf.astro_object import (  # ztf parser moved, names re-exported for step.py and tests
    add_mag_and_flux_columns,  # ztf magnitude and flux rows
    detections_to_astro_object,  # ztf epochs -> AstroObject
    get_bogus_flags_for_each_detection,  # rb and procstatus per epoch
    get_new_references_from_message,  # references carried in the message
    get_reference_for_each_detection,  # distnr and rfid per epoch
)  # end of the ztf parser imports
from features.surveys.ztf.output import parse_output  # ztf output messages, re-exported
from features.surveys.ztf.scribe import fid_mapper_for_db, parse_scribe_payload  # ztf scribe, re-exported


def flux_err_2_mag_err(flux_err, flux):
    return (2.5 * flux_err) / (np.log(10.0) * flux)

def fluxnjy2mag(flux):
    return 31.4 - 2.5 * np.log10(flux)


def add_mag_and_flux_lsst(a: pd.DataFrame) -> pd.DataFrame:
    """
    Lógica de magnitud y flujo para LSST.
    """
    a_flux = a.copy()
    # Filter to keep only rows with positive scienceFlux and scienceFluxErr
    a = a[(a['scienceFlux'] > 0) & (a['scienceFluxErr'] > 0)].copy()

    a['scienceFluxErr'] = a.apply(lambda row: flux_err_2_mag_err(row['scienceFluxErr'], abs(row['scienceFlux'])), axis=1)
    a['scienceFlux'] = a['scienceFlux'].apply(fluxnjy2mag)

    a.rename(
        columns={"scienceFlux": "brightness", "scienceFluxErr": "e_brightness"},
        inplace=True,
    )
    
    a["unit"] = "magnitude"
    a_flux["brightness"] = a_flux["psfFlux"]/1000 #njy
    a_flux["e_brightness"] = a_flux["psfFluxErr"]/1000
    a_flux["unit"] = "diff_flux"
    a = pd.concat([a, a_flux], axis=0,ignore_index=True)
    for col in ["psfFlux", "psfFluxErr","scienceFlux","scienceFluxErr"]:
        if col in a.columns:
            a.drop(columns=[col], inplace=True)

    return a

def detections_to_astro_object_lsst(
    detections: list,
    forced: list,
    xmatches: dict = None,
    references_db: pd.DataFrame = None,
) -> AstroObject:
    
    detection_keys = ["oid","sid","mjd","ra","dec",
                      "psfFlux","psfFluxErr","scienceFlux",
                      "scienceFluxErr","tid","band","measurement_id"]
    
    # Build values list with forced flag included from the start
    values = []
    
    # Process regular detections (forced=False)
    for detection in detections:
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]
        row.append(False)  # forced = False
        values.append(row)
    
    # Process forced photometry (forced=True)  
    for detection in forced:
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]
        row.append(True)  # forced = True
        values.append(row)

    a = pd.DataFrame(data=values, columns=detection_keys + ['forced'])
    a.fillna(value=np.nan, inplace=True)
    # Renombrar las columnas de 'a' de acuerdo con DETECTION_KEYS_MAP
    a.rename(columns=DETECTION_KEYS_MAP, inplace=True)
    
    band_map = {"g": 1, "r": 2, "i": 3, "z": 4, "y": 5, "u": 6}
    band_map_inverse = {v: k for k, v in band_map.items()}
    a["fid"] = a["fid"].map(band_map_inverse)

    a = add_mag_and_flux_lsst(a)

    oid = a["oid"].iloc[0]

    aid_forced = a[a["forced"]]
    aid_detections = a[~a["forced"]]

    w1, w2, w3, w4 = wise_magnitudes(xmatches)  # W1-W4 from an allwise match, else NaN

    last_mjd = last_mjd_of(detections, forced)  # max mjd over detections and forced photometry


    metadata = pd.DataFrame(
        [   ["aid","aid"],
            ["oid", oid],
            ["W1", w1],
            ["W2", w2],
            ["W3", w3],
            ["W4", w4],
            ["last_mjd", last_mjd],
        ],
        columns=["name", "value"],)
    
    astro_object = AstroObject(
        detections=aid_detections, #detections deberian mantenerse en njy 
        forced_photometry=aid_forced,
        metadata=metadata,
        reference=None,
    )
    return astro_object

def fid_mapper_for_db_lsst(band: str) -> int:
    """
    Map LSST band identifiers to DB fid codes using explicit mapping.

    Singles: u,g,r,i,z,y -> 6,1,2,3,4,5 #esto cambiar
    Valid combinations: u,g g,r r,i i,z z,y -> 61,12,23,34,45
    """
    if band is None:
        return 0 #reemplazamos valor nulo por 0
    band = str(band).strip()
    band_to_fid = {
        "u": 6, #0 
        "g": 1, #1
        "r": 2,
        "i": 3,
        "z": 4,
        "y": 5,
        "u,g": 61, # 10
        "g,r": 12,
        "r,i": 23,
        "i,z": 34,
        "z,y": 45,
    }
    return band_to_fid.get(band, 0)


def prepare_ao_features_for_db(astro_object: AstroObject, feature_name_lut) -> pd.DataFrame:
    """Return the ztf feature rows for the scribe upsert, keeping the name column."""  # docstring of the delegation
    return prepare_features_for_db(astro_object, feature_name_lut, ZTFSurvey.bands, keep_name=True)  # shared body, ztf bands


def prepare_ao_features_for_db_lsst(astro_object: AstroObject, feature_name_lut) -> pd.DataFrame:
    """Return the lsst feature rows for the scribe upsert, without the name column."""  # docstring of the delegation
    return prepare_features_for_db(astro_object, feature_name_lut, LSSTSurvey.bands, keep_name=False)  # shared body, lsst bands


def parse_scribe_payload_lsst(
    astro_objects: List[AstroObject], features_version, features_group, feature_name_lut
):
    """Create scribe commands for LSST without color updates.

    Builds only update_features commands and returns them under
    the 'upserting_features' key. No update_object commands are
    created or returned.
    """
    upsert_features_commands_list = []

    for astro_object in astro_objects:
        ao_features = prepare_ao_features_for_db_lsst(astro_object,feature_name_lut)
        oid = query_ao_table(astro_object.metadata, "oid")
        last_mjd = query_ao_table(astro_object.metadata, "last_mjd")


        features_list = ao_features.to_dict("records")
        sid = astro_object.detections["sid"].values[0]
        # New envelope format required downstream
        upsert_features_command = {
            "step": "features", #cual tiene que ser el nombre aqui?
            "survey": "lsst",
            "payload": {
                "oid": int(oid),
                "features_version": features_version,
                "sid": sid,
                "mjd": last_mjd,
                "features": features_list,
            },
        }
        upsert_features_commands_list.append(upsert_features_command)

    return {"upserting_features": upsert_features_commands_list}

def parse_output_lsst(
    astro_objects: List[AstroObject],
    messages: List[Dict],
    candids: Dict,
) -> list[dict]:
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
    output_messages: list[dict] = []

    # LSST band -> suffix index mapping
    suffix_map = { #u = 6
        "u": "_6", "g": "_1", "r": "_2", "i": "_3", "z": "_4", "y": "_5",
        "u,g": "_61", "g,r": "_12", "r,i": "_23", "i,z": "_34", "z,y": "_45",
        None: "",
    }

    for message, astro_object in zip(messages, astro_objects):
        oid = message["oid"]
        measurement_id = candids[oid]

        # Ensure AO metadata oid matches message oid
        oid_ao = query_ao_table(astro_object.metadata, "oid") #es esto necesario?
        assert oid_ao == oid

        ao_features = astro_object.features[["name", "fid", "value"]].copy()
        # Append band suffix to feature name
        ao_features["name"] += ao_features["fid"].map(suffix_map).fillna("")
        ao_features = ao_features.sort_values("name")
        ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)

        # Sanitize feature names
        feature_names = [f.replace("-", "_") for f in ao_features["name"].values]
        features_for_oid = dict(
            zip(feature_names, ao_features["value"].astype(float))
        )
        for key in list(features_for_oid.keys()):
            val = features_for_oid[key]
            features_for_oid[key] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else val

        out_message = {
            "oid": oid,
            "measurement_id": measurement_id,
            "detections": message.get("detections", []), #nyj 
            "features": features_for_oid, #features fueron calculados con microjy
            #features en que unidades quedaran.
        }
        output_messages.append(out_message)

    return output_messages
