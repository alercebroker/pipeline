import logging
import numpy as np
import pandas as pd
from lc_classifier.features.core.base import AstroObject, query_ao_table
from lc_classifier.utils import mag2flux, mag_err_2_flux_err
from typing import List, Dict, Optional


DETECTION_KEYS_MAP = {
    "oid": "oid",
    "sid": "sid",
    "mjd": "mjd",
    "ra": "ra",
    "dec": "dec",
    "tid": "tid",
    "band": "band",  # LSST usa 'band', ZTF usa 'fid'
    "diaObjectId": "diaObjectId",  # LSST 'diaObjectId' ≈ ZTF 'candid'
    "candid": "diaObjectId",
    "aid": "aid",
    "fid": "band",  # ZTF 'fid' ≈ LSST 'band'
    "isdiffpos": "isdiffpos",
    "forced": "forced",
    "pid": "pid"
}


def add_mag_and_flux_lsst(a: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder para lógica de magnitud y flujo para LSST.
    """
    # Aquí irá la lógica específica para LSST
    #aqui tengo que terminar con las columnas brigthness y unit

    a.rename(
        columns={"psfFlux": "brightness", "psfFluxErr": "e_brightness"},
        inplace=True,
    )

    a["unit"] = "magnitude"

    a_flux = a.copy()
    # TODO: check this
    a_flux["e_brightness"] = a["scienceFlux"]#mag_err_2_flux_err(a["e_mag"], a["mag"])#esto preguntar
    a_flux["brightness"] = a["scienceFluxErr"]#mag2flux(a["mag"]) * a["isdiffpos"] #esto preguntar
    a_flux["unit"] = "diff_flux"
    a = pd.concat([a, a_flux], axis=0)
    a.set_index("aid", inplace=True)
    for col in ["mag", "e_mag"]:
        if col in a.columns:
            a.drop(columns=[col], inplace=True)
    return a

def detections_to_astro_object_lsst(
    detections: list,
    forced: list,
    xmatches: dict = None,
    references_db: pd.DataFrame = None,
) -> AstroObject:
    
    #lo primero es estandarizar los nombres. Voy a hacer eso. Voy a revisar el schema de correction y ocupar las claves minimas.

    #psfFlux es mag_corr
    #scienceFlux es mag
    detection_keys = ["oid","sid","mjd","ra","dec",
                      "psfFlux","psfFluxErr","scienceFlux",
                      "scienceFluxErr","tid","band","diaObjectId"]
    
    #quiero estandarizar las detection keys con la otra funcion




    #pid no va
    #fid es band
    #isdiffpos no va
    #forced se trata por separado
    #aid no se usa creo
    #candid que es?
    #diaobjectid es candid entonces
    #detection_keys = [
    #     "candid", "aid", "forced"
    #]

    values = []
    for detection in detections:
        values.append([detection.get(key, None) for key in detection_keys])

    a = pd.DataFrame(data=values, columns=detection_keys)
    a.fillna(value=np.nan, inplace=True)

    a = add_mag_and_flux_lsst(a)
    
    aid = a.index.values[0]
    oid = a["oid"].iloc[0]

    aid_forced = None#a[a["forced"]] #que hacer con las forced?
    aid_detections = a#a[~a["forced"]]

    astro_object = AstroObject(
        detections=aid_detections,
        forced_photometry=aid_forced,
        metadata=None,
        reference=None,
    )
    return astro_object
def add_mag_and_flux_columns(a: pd.DataFrame) -> pd.DataFrame:
    """
    Toma un DataFrame con columnas 'mag', 'e_mag', 'mag_corr', 'e_mag_corr_ext', 'isdiffpos', 'forced',
    y retorna un DataFrame con columnas de magnitud y flujo, renombradas y calculadas.
    """
    a = a[(a["mag"] != 100) | (a["e_mag"] != 100)].copy()
    a.rename(
        columns={"mag_corr": "brightness", "e_mag_corr_ext": "e_brightness"},
        inplace=True,
    )
    a["unit"] = "magnitude"
    a_flux = a.copy()
    # TODO: check this
    a_flux["e_brightness"] = mag_err_2_flux_err(a["e_mag"], a["mag"])
    a_flux["brightness"] = mag2flux(a["mag"]) * a["isdiffpos"]
    a_flux["unit"] = "diff_flux"
    a = pd.concat([a, a_flux], axis=0)
    a.set_index("aid", inplace=True)
    for col in ["mag", "e_mag"]:
        if col in a.columns:
            a.drop(columns=[col], inplace=True)
    return a



def get_bogus_flags_for_each_detection(detections: List[Dict]):
    # for each detection, it looks for the real-bogus score (available only for
    # detections) and procstatus flag (available only for forced
    # photometry epochs)

    keys = ["rb", "procstatus"]

    bogus_flags = []
    for detection in detections:
        value = []
        for key in keys:
            if key in detection["extra_fields"].keys():
                value.append(detection["extra_fields"][key])
            else:
                value.append(None)
        bogus_flags.append(value)

    bogus_flags = pd.DataFrame(bogus_flags, columns=keys)
    bogus_flags["procstatus"] = bogus_flags["procstatus"].astype(str)

    return bogus_flags


def get_reference_for_each_detection(detections: List[Dict]):
    # for each detection, it looks what is the reference id
    # and how far away it is

    keys = ["distnr", "rfid"]

    reference = []
    for detection in detections:
        value = []
        for key in keys:
            if key in detection["extra_fields"].keys():
                value.append(detection["extra_fields"][key])
            else:
                value.append(None)
        reference.append(value)

    reference = pd.DataFrame(reference, columns=keys)

    return reference


def get_new_references_from_message(detections: List[Dict]) -> pd.DataFrame:
    # get info of references that is in the incoming message
    #
    # output columns: oid, rfid, sharpnr, chinr

    keys = ["rfid", "sharpnr", "chinr"]

    references = []
    for detection in detections:
        if not set(keys).issubset(detection["extra_fields"]):
            continue
        reference = [detection["oid"]] + [detection["extra_fields"][k] for k in keys]
        references.append(reference)

    references = pd.DataFrame(references, columns=["oid"] + keys)
    references = references[references["chinr"] >= 0.0].copy()
    references.drop_duplicates(["oid", "rfid"], keep="first", inplace=True)
    return references


def detections_to_astro_object(
    detections: List[Dict],
    xmatches: Optional[Dict],
    references_db: Optional[pd.DataFrame],
) -> AstroObject:
    detection_keys = [
        "oid",
        "candid",
        "aid",
        "tid",
        "sid",
        "pid",
        "ra",
        "dec",
        "mjd",
        "mag_corr",
        "e_mag_corr_ext",
        "mag",
        "e_mag",
        "fid",
        "isdiffpos",
        "forced",
    ]

    values = []
    for detection in detections:
        values.append([detection[key] for key in detection_keys])

    a = pd.DataFrame(data=values, columns=detection_keys)
    a.fillna(value=np.nan, inplace=True)

    # reference_for_each_detection has distnr, rfid from dets
    reference_for_each_detection: pd.DataFrame = get_reference_for_each_detection(
        detections
    )
    a = pd.concat([a, reference_for_each_detection], axis=1)

    bogus_flags_for_each_detection: pd.DataFrame = get_bogus_flags_for_each_detection(
        detections
    )
    a = pd.concat([a, bogus_flags_for_each_detection], axis=1)

    a = add_mag_and_flux_columns(a)

    aid = a.index.values[0]
    oid = a["oid"].iloc[0]

    aid_forced = a[a["forced"]]
    aid_detections = a[~a["forced"]]

    w1 = w2 = w3 = w4 = np.nan
    if xmatches is not None and "allwise" in xmatches.keys():
        w1 = xmatches["allwise"]["W1mag"]
        w2 = xmatches["allwise"]["W2mag"]
        w3 = xmatches["allwise"]["W3mag"]
        w4 = xmatches["allwise"]["W4mag"]

    sgscore1 = np.nan
    sgmag1 = np.nan
    srmag1 = np.nan
    simag1 = np.nan
    szmag1 = np.nan
    distpsnr1 = np.nan

    for det in detections:
        if "sgscore1" in det["extra_fields"].keys():
            sgscore1 = det["extra_fields"]["sgscore1"]
            sgmag1 = det["extra_fields"]["sgmag1"]
            srmag1 = det["extra_fields"]["srmag1"]
            simag1 = det["extra_fields"]["simag1"]
            szmag1 = det["extra_fields"]["szmag1"]
            distpsnr1 = det["extra_fields"]["distpsnr1"]
            continue

    metadata = pd.DataFrame(
        [
            ["aid", aid],
            ["oid", oid],
            ["W1", w1],
            ["W2", w2],
            ["W3", w3],
            ["W4", w4],
            ["sgscore1", sgscore1],
            ["sgmag1", sgmag1],
            ["srmag1", srmag1],
            ["simag1", simag1],
            ["szmag1", szmag1],
            ["distpsnr1", distpsnr1],
        ],
        columns=["name", "value"],
    ).fillna(value=np.nan)

    new_references = get_new_references_from_message(detections)

    if references_db is not None:
        references_db = references_db[references_db["oid"] == oid].copy()
        references = pd.concat([new_references, references_db], axis=0)
        references.drop_duplicates(["oid", "rfid"], keep="first", inplace=True)
    else:
        references = new_references

    astro_object = AstroObject(
        detections=aid_detections,
        forced_photometry=aid_forced,
        metadata=metadata,
        reference=references,
    )
    return astro_object


def fid_mapper_for_db(band: str):
    """
    Parses the number used to reference the fid in the ztf alerts
    to the string value corresponding
    """
    fid_map = {"g": 1, "r": 2, "g,r": 12}
    if band in fid_map:
        return fid_map[band]
    return 0


def prepare_ao_features_for_db(astro_object: AstroObject) -> pd.DataFrame: #esto tengo que verlo
    ao_features = astro_object.features[["name", "fid", "value"]].copy()
    ao_features["fid"] = ao_features["fid"].apply(fid_mapper_for_db)
    ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)

    # backward compatibility
    ao_features["name"] = ao_features["name"].replace(
        {
            "Power_rate_1_4": "Power_rate_1/4",
            "Power_rate_1_3": "Power_rate_1/3",
            "Power_rate_1_2": "Power_rate_1/2",
        }
    )
    return ao_features


def parse_scribe_payload(
    astro_objects: List[AstroObject], features_version, features_group
):
    """Create the json with the messages for the scribe producer from the
    features dataframe. It adds the fid and correct the name.

    :param astro_objects: a list of AstroObjects with computed features inside.
    :param features_version: a string with the features version used
    :return: a list of json with Alerce Scribe commands
    """

    # features = features.replace({np.nan: None, np.inf: None, -np.inf: None})
    upsert_features_commands_list = []
    update_object_command_list = []

    for astro_object in astro_objects:
        # for upserting features
        ao_features = prepare_ao_features_for_db(astro_object)
        oid = query_ao_table(astro_object.metadata, "oid")

        features_list = ao_features.to_dict("records")

        upsert_features_command = {
            "collection": "object",
            "type": "update_features",
            "criteria": {"_id": oid},
            "data": {
                "features_version": features_version,
                "features_group": features_group,
                "features": features_list,
            },
            "options": {"upsert": True},
        }
        upsert_features_commands_list.append(upsert_features_command)

        # for updating the object
        def get_color_from_features(name, features_list):
            color = list(
                filter(lambda x: x["name"] == name and x["fid"] == 12, features_list)
            )
            color = color[0]["value"] if len(color) == 1 else None
            return color

        update_object_command = {
            "collection": "object",
            "type": "update_object_from_stats",
            "criteria": {"oid": oid},
            "data": {
                "g_r_max": get_color_from_features("g_r_max", features_list),
                "g_r_mean": get_color_from_features("g_r_mean", features_list),
                "g_r_max_corr": get_color_from_features("g_r_max_corr", features_list),
                "g_r_mean_corr": get_color_from_features(
                    "g_r_mean_corr", features_list
                ),
            },
            "options": {},
        }
        update_object_command_list.append(update_object_command)

    return {
        "update_object": update_object_command_list,
        "upserting_features": upsert_features_commands_list,
    }


def parse_output(
    astro_objects: List[AstroObject], messages: List[Dict], candids: Dict
) -> list[dict]: #esto igual tengo que verlo
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

    output_messages = []
    for message, astro_object in zip(messages, astro_objects):
        oid = message["oid"]
        candid = candids[oid]

        ao_features = astro_object.features[["name", "fid", "value"]].copy()
        fid_map = {"g": "_1", "r": "_2", "g,r": "_12", None: ""}
        ao_features["name"] += ao_features["fid"].map(fid_map)
        ao_features = ao_features.sort_values("name")
        ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
        oid_ao = query_ao_table(astro_object.metadata, "oid")
        assert oid_ao == oid
        feature_names = [f.replace("-", "_") for f in ao_features["name"].values]

        reference = astro_object.reference
        if reference is not None:
            reference = astro_object.reference.reset_index(drop=True).to_dict("records")

        features_for_oid = dict(
            zip(feature_names, ao_features["value"].astype(np.double))
        )
        for key in features_for_oid.keys():
            features_for_oid[key] = (
                None if np.isnan(features_for_oid[key]) else features_for_oid[key]
            )

        out_message = {
            "oid": oid,
            "candid": candid,
            "detections": message["detections"],
            "non_detections": message["non_detections"],
            "xmatches": message["xmatches"],
            "features": features_for_oid,
            "reference": reference,
        }
        output_messages.append(out_message)

    return output_messages
