"""lsst message epochs -> AstroObject, moved from features/utils/parsers.py."""
import numpy as np  # NaN placeholders and logarithms
import pandas as pd  # epoch frames
from lc_classifier.features.core.base import AstroObject  # object the parser builds

from features.surveys.common import DETECTION_KEYS_MAP, last_mjd_of, wise_magnitudes  # shared survey helpers
from features.surveys.lsst.bands import LSST_BANDS  # lsst band table


def flux_err_2_mag_err(flux_err, flux):
    """Return the magnitude error of `flux` given its error `flux_err`."""
    return (2.5 * flux_err) / (np.log(10.0) * flux)  # first-order propagation of the flux error

def fluxnjy2mag(flux):
    """Return the AB magnitude of a flux in nJy."""
    return 31.4 - 2.5 * np.log10(flux)  # 31.4 is the AB zero point for nJy


def add_mag_and_flux_lsst(a: pd.DataFrame) -> pd.DataFrame:
    """
    Lógica de magnitud y flujo para LSST.
    """
    a_flux = a.copy()  # every row, kept for the flux half
    # Filter to keep only rows with positive scienceFlux and scienceFluxErr
    a = a[(a['scienceFlux'] > 0) & (a['scienceFluxErr'] > 0)].copy()  # magnitude half: positive science flux and error

    a['scienceFluxErr'] = a.apply(lambda row: flux_err_2_mag_err(row['scienceFluxErr'], abs(row['scienceFlux'])), axis=1)  # magnitude error
    a['scienceFlux'] = a['scienceFlux'].apply(fluxnjy2mag)  # science flux -> magnitude

    a.rename(  # science columns become the brightness columns
        columns={"scienceFlux": "brightness", "scienceFluxErr": "e_brightness"},  # old -> new names
        inplace=True,  # rename in place
    )  # end of the rename

    a["unit"] = "magnitude"  # these rows are magnitudes
    a_flux["brightness"] = a_flux["psfFlux"]/1000 #njy
    a_flux["e_brightness"] = a_flux["psfFluxErr"]/1000  # difference flux error, same scaling
    a_flux["unit"] = "diff_flux"  # these rows are fluxes
    a = pd.concat([a, a_flux], axis=0,ignore_index=True)  # magnitude rows then flux rows, new RangeIndex
    for col in ["psfFlux", "psfFluxErr","scienceFlux","scienceFluxErr"]:  # raw flux columns
        if col in a.columns:  # only when present
            a.drop(columns=[col], inplace=True)  # no longer needed

    return a  # magnitude and flux rows

def detections_to_astro_object_lsst(  # signature spans several lines
    detections: list,  # sources and previous sources that carry a band
    forced: list,  # forced sources of the message
    xmatches: dict = None,  # allwise match or None
    references_db: pd.DataFrame = None,  # accepted and not read: lsst builds no reference frame
) -> AstroObject:  # returns the lsst AstroObject
    """Return the lsst AstroObject built from the message sources, the forced sources and the xmatch."""

    detection_keys = ["oid","sid","mjd","ra","dec",  # input keys copied into the epoch frame
                      "psfFlux","psfFluxErr","scienceFlux",  # difference flux, its error, science flux
                      "scienceFluxErr","tid","band","measurement_id"]  # science flux error, telescope, band, detection id

    # Build values list with forced flag included from the start
    values = []  # one row per epoch

    # Process regular detections (forced=False)
    for detection in detections:  # every source
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]  # sid as str
        row.append(False)  # forced = False
        values.append(row)  # add the row

    # Process forced photometry (forced=True)
    for detection in forced:  # every forced source
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]  # sid as str
        row.append(True)  # forced = True
        values.append(row)  # add the row

    a = pd.DataFrame(data=values, columns=detection_keys + ['forced'])  # rows -> frame
    a.fillna(value=np.nan, inplace=True)  # None -> NaN
    # Renombrar las columnas de 'a' de acuerdo con DETECTION_KEYS_MAP
    a.rename(columns=DETECTION_KEYS_MAP, inplace=True)  # band -> fid, measurement_id -> candid

    a["fid"] = a["fid"].map(LSST_BANDS.singles_inverse)  # numeric band -> band letter

    a = add_mag_and_flux_lsst(a)  # magnitude and flux rows

    oid = a["oid"].iloc[0]  # oid of the object

    aid_forced = a[a["forced"]]  # forced photometry rows
    aid_detections = a[~a["forced"]]  # detection rows

    w1, w2, w3, w4 = wise_magnitudes(xmatches)  # W1-W4 from an allwise match, else NaN

    last_mjd = last_mjd_of(detections, forced)  # max mjd over detections and forced photometry


    metadata = pd.DataFrame(  # metadata frame
        [   ["aid","aid"],  # the literal string "aid", as today
            ["oid", oid],  # object id
            ["W1", w1],  # allwise W1
            ["W2", w2],  # allwise W2
            ["W3", w3],  # allwise W3
            ["W4", w4],  # allwise W4
            ["last_mjd", last_mjd],  # newest epoch
        ],  # end of the rows
        columns=["name", "value"],)  # name/value layout

    astro_object = AstroObject(  # the lsst AstroObject
        detections=aid_detections, #detections deberian mantenerse en njy
        forced_photometry=aid_forced,  # forced photometry rows
        metadata=metadata,  # name/value metadata
        reference=None,  # lsst has no references
    )  # end of the AstroObject
    return astro_object  # ready for the preprocessor
