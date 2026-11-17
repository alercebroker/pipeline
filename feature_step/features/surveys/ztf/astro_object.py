"""ztf message epochs -> AstroObject, moved from features/utils/parsers.py."""
from typing import Dict, List, Optional  # type hints of the moved signatures

import numpy as np  # NaN placeholders
import pandas as pd  # epoch frames
from lc_classifier.features.core.base import AstroObject  # object the parser builds
from lc_classifier.utils import mag2flux, mag_err_2_flux_err  # magnitude -> flux conversions

from features.surveys.common import DETECTION_KEYS_MAP, last_mjd_of, wise_magnitudes  # shared survey helpers
from features.surveys.ztf.bands import ZTF_BANDS  # ztf band table


def add_mag_and_flux_columns(a: pd.DataFrame) -> pd.DataFrame:
    """
    Toma un DataFrame con columnas 'mag', 'e_mag', 'magpsf_corr', 'sigmapsf_corr_ext', 'isdiffpos', 'forced',
    y retorna un DataFrame con columnas de magnitud y flujo, renombradas y calculadas.
    """
    a = a[(a["mag"] != 100) | (a["e_mag"] != 100)].copy()  # drop epochs with the 100 placeholder magnitude
    a.rename(  # corrected magnitude becomes the brightness columns
        columns={"magpsf_corr": "brightness", "sigmapsf_corr_ext": "e_brightness"},  # old -> new names
        inplace=True,  # rename in place
    )  # end of the rename
    a["unit"] = "magnitude"  # these rows are magnitudes
    a_flux = a.copy()  # second copy for the flux rows
    # TODO: check this
    a_flux["e_brightness"] = mag_err_2_flux_err(a["e_mag"], a["mag"])  # flux error from the magnitude error
    a_flux["brightness"] = mag2flux(a["mag"]) * a["isdiffpos"]  # signed difference flux
    a_flux["unit"] = "diff_flux"  # these rows are fluxes
    a = pd.concat([a, a_flux], axis=0)  # magnitude rows then flux rows
    a.set_index("aid", inplace=True)  # AstroObject frames are indexed by aid
    for col in ["mag", "e_mag"]:  # raw magnitude columns
        if col in a.columns:  # only when present
            a.drop(columns=[col], inplace=True)  # no longer needed
    return a  # magnitude and flux rows



def get_bogus_flags_for_each_detection(detections: List[Dict]):
    """Return a frame with rb and procstatus (as str) for each detection, None when absent."""
    # for each detection, it looks for the real-bogus score (available only for
    # detections) and procstatus flag (available only for forced
    # photometry epochs)

    keys = ["rb", "procstatus"]  # flags to read

    bogus_flags = []  # one row per detection
    for detection in detections:  # every epoch
        value = []  # row of this epoch
        for key in keys:  # rb then procstatus
            if key in detection.keys():  # flag present
                value.append(detection[key])  # keep its value
            else:  # flag absent
                value.append(None)  # missing marker
        bogus_flags.append(value)  # add the row

    # Stringify from the original values, not from the built column: procstatus
    # may arrive as an int, and a column mixing ints with None becomes float64,
    # so .astype(str) would render 0 as "0.0" and discard every forced epoch.
    # Assumes procstatus is always str or int, never a genuine Python float
    # (e.g. 0.0) -- true today since the Avro schema types it ["null","string"];
    # a real float would still stringify as "0.0" and be wrongly discarded.
    procstatus = [str(row[keys.index("procstatus")]) for row in bogus_flags]  # procstatus as str

    bogus_flags = pd.DataFrame(bogus_flags, columns=keys)  # rows -> frame
    bogus_flags["procstatus"] = procstatus  # overwrite with the str values

    return bogus_flags  # rb and procstatus per epoch


def get_reference_for_each_detection(detections: List[Dict]):
    """Return a frame with distnr and rfid for each detection, None when absent."""
    # for each detection, it looks what is the reference id
    # and how far away it is

    keys = ["distnr", "rfid"]  # reference columns to read

    reference = []  # one row per detection
    for detection in detections:  # every epoch
        value = []  # row of this epoch
        for key in keys:  # distnr then rfid
            if key in detection.keys():  # key present
                value.append(detection[key])  # keep its value
            else:  # key absent
                value.append(None)  # missing marker
        reference.append(value)  # add the row

    reference = pd.DataFrame(reference, columns=keys)  # rows -> frame

    return reference  # distnr and rfid per epoch


def get_new_references_from_message(detections: List[Dict]) -> pd.DataFrame:
    """Return the references (oid, rfid, sharpnr, chinr) carried by the message epochs."""
    # get info of references that is in the incoming message
    #
    # output columns: oid, rfid, sharpnr, chinr

    keys = ["rfid", "sharpnr", "chinr"]  # reference columns to read

    references = []  # one row per epoch that carries a reference
    for detection in detections:  # every epoch
        if not set(keys).issubset(detection):  # epoch without the full reference
            continue  # skip it
        reference = [detection["oid"]] + [detection[k] for k in keys]  # oid then reference values
        references.append(reference)  # add the row

    references = pd.DataFrame(references, columns=["oid"] + keys)  # rows -> frame
    references = references[references["chinr"] >= 0.0].copy()  # same chinr filter as the db query
    references.drop_duplicates(["oid", "rfid"], keep="first", inplace=True)  # one row per reference
    return references  # references from the message


def detections_to_astro_object(  # signature spans several lines
    detections: List[Dict],  # epochs: detections, previous detections and forced photometry inline
    forced: List[Dict],  # must be empty for ztf
    xmatches: Optional[Dict],  # allwise match or None
    references_db: Optional[pd.DataFrame],  # references read from the db, or None
) -> AstroObject:  # returns the ztf AstroObject
    """Return the ztf AstroObject built from the message epochs, the xmatch and the references."""
    detection_keys = [  # input keys copied into the epoch frame
        "oid",  # object id
        "measurement_id",  # detection id
        "aid",  # alerce id, stamped from oid
        "tid",  # telescope id
        "sid",  # survey id, stored as str
        "pid",  # processing id
        "ra",  # right ascension
        "dec",  # declination
        "mjd",  # epoch time
        "magpsf_corr",      # candidate / prv_candidate spelling
        "mag_corr",         # forced_photometry spelling
        "sigmapsf_corr_ext",  # candidate / prv_candidate error spelling
        "e_mag_corr_ext",  # forced_photometry error spelling
        "mag",  # raw magnitude
        "e_mag",  # raw magnitude error
        "band",  # numeric band
        "isdiffpos",  # sign of the difference flux
        "forced",  # forced photometry flag
    ]  # end of the keys

    # Forced epochs arrive inline in `detections` with a per-row `forced` flag,
    # so this arg must be empty. A non-empty `forced` would misalign the
    # column-wise concat of the reference/bogus frames, which are computed over
    # `detections` alone.
    forced = forced or []  # None counts as empty
    if forced:  # forced epochs passed separately
        raise NotImplementedError(  # refuse: they would misalign the concat
            "detections_to_astro_object: `forced` must be empty for ZTF; forced "  # message, first half
            "epochs flow inline via the per-row `forced` flag in `detections`."  # message, second half
        )  # end of the raise

    values = []  # one row per epoch
    for detection in detections:  # every epoch
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]  # sid as str
        values.append(row)  # add the row

    a = pd.DataFrame(data=values, columns=detection_keys)  # rows -> frame
    a.fillna(value=np.nan, inplace=True)  # None -> NaN

    # A record's type selects at most one spelling of the corrected magnitude --
    # the other key is absent from its schema entirely -- so the coalesce can
    # never have to choose. An uncorrected epoch populates neither and stays NaN.
    # This must run before the DETECTION_KEYS_MAP rename.
    a["magpsf_corr"] = a["magpsf_corr"].fillna(a["mag_corr"])  # coalesce the corrected magnitude
    a["sigmapsf_corr_ext"] = a["sigmapsf_corr_ext"].fillna(a["e_mag_corr_ext"])  # coalesce its error
    a.drop(columns=["mag_corr", "e_mag_corr_ext"], inplace=True)  # forced spellings no longer needed

    # reference_for_each_detection has distnr, rfid from dets
    reference_for_each_detection: pd.DataFrame = get_reference_for_each_detection(  # distnr and rfid per epoch
        detections  # same list as the epoch frame
    )  # end of the call
    bogus_flags_for_each_detection: pd.DataFrame = get_bogus_flags_for_each_detection(  # rb and procstatus per epoch
        detections  # same list as the epoch frame
    )  # end of the call

    # All three frames come from the same unfiltered list in the same order, so
    # they share a RangeIndex and concat aligns. Assert it: a length mismatch
    # here NaN-fills distnr/rfid/rb/procstatus silently rather than failing,
    # which is the bug this single-loop form removed.
    assert len(a) == len(reference_for_each_detection) == len(  # same row count in the three frames
        bogus_flags_for_each_detection  # third frame
    )  # end of the assert

    a = pd.concat([a, reference_for_each_detection], axis=1)  # add distnr and rfid
    a = pd.concat([a, bogus_flags_for_each_detection], axis=1)  # add rb and procstatus

    a.rename(columns=DETECTION_KEYS_MAP, inplace=True)  # band -> fid, measurement_id -> candid

    a = add_mag_and_flux_columns(a)  # magnitude and flux rows, indexed by aid

    aid = a.index.values[0]  # aid of the object
    oid = a["oid"].iloc[0]  # oid of the object

    a["fid"] = a["fid"].map(ZTF_BANDS.singles_inverse)  # numeric fid -> band letter

    aid_forced = a[a["forced"]]  # forced photometry rows
    aid_detections = a[~a["forced"]]  # detection rows


    w1, w2, w3, w4 = wise_magnitudes(xmatches)  # W1-W4 from an allwise match, else NaN


    sgscore1 = np.nan  # star-galaxy score, NaN until an epoch carries it
    sgmag1 = np.nan  # PS1 g magnitude
    srmag1 = np.nan  # PS1 r magnitude
    simag1 = np.nan  # PS1 i magnitude
    szmag1 = np.nan  # PS1 z magnitude
    distpsnr1 = np.nan  # distance to the PS1 source

    for det in detections:  # every epoch
        if "sgscore1" in det.keys():  # epoch carries the PS1 match
            sgscore1 = det["sgscore1"]  # star-galaxy score
            sgmag1 = det["sgmag1"]  # PS1 g magnitude
            srmag1 = det["srmag1"]  # PS1 r magnitude
            simag1 = det["simag1"]  # PS1 i magnitude
            szmag1 = det["szmag1"]  # PS1 z magnitude
            distpsnr1 = det["distpsnr1"]  # distance to the PS1 source
            continue  # the last epoch with a match wins

    last_mjd = last_mjd_of(detections, forced)  # max mjd over detections and forced photometry

    metadata_rows = [  # name, value rows of the metadata frame
        ["aid", aid],  # alerce id
        ["oid", oid],  # object id
        ["W1", w1],  # allwise W1
        ["W2", w2],  # allwise W2
        ["W3", w3],  # allwise W3
        ["W4", w4],  # allwise W4
        ["sgscore1", sgscore1],  # star-galaxy score
        ["sgmag1", sgmag1],  # PS1 g magnitude
        ["srmag1", srmag1],  # PS1 r magnitude
        ["simag1", simag1],  # PS1 i magnitude
        ["szmag1", szmag1],  # PS1 z magnitude
        ["distpsnr1", distpsnr1],  # distance to the PS1 source
        ["last_mjd", last_mjd],  # newest epoch
    ]  # end of the rows
    # dtype=object keeps `oid`/`aid` as exact ints: ZTF multisurvey oids exceed
    # 2**53, so a float64 column silently rounds them. None -> NaN is done here
    # rather than with .fillna, which would downcast the column back to float64.
    metadata = pd.DataFrame(  # metadata frame
        [[name, np.nan if value is None else value] for name, value in metadata_rows],  # None -> NaN
        columns=["name", "value"],  # name/value layout
        dtype=object,  # keep exact ints
    )  # end of the frame

    new_references = get_new_references_from_message(detections)  # references in the message

    if references_db is not None:  # references read from the db
        references_db = references_db[references_db["oid"] == oid].copy()  # this object only
        references = pd.concat([new_references, references_db], axis=0)  # message first, then db
        references.drop_duplicates(["oid", "rfid"], keep="first", inplace=True)  # message wins on duplicates
    else:  # no db references
        references = new_references  # message references only

    astro_object = AstroObject(  # the ztf AstroObject
        detections=aid_detections,  # detection rows
        forced_photometry=aid_forced,  # forced photometry rows
        metadata=metadata,  # name/value metadata
        reference=references,  # references of the object
    )  # end of the AstroObject
    return astro_object  # ready for the preprocessor
