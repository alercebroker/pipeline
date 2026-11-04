"""Build synthetic LSST multisurvey messages for the feature step, without the `jsons/` dir.

Each epoch carries exactly the keys `detections_to_astro_object_lsst` reads.
Three edge rows are pinned: a source with `band: None` (dropped in pre_execute),
a source with `scienceFlux: -5.0` (dropped from the magnitude half) and a
non-empty `forced_sources`.
"""
import random  # seeded generator so every message is reproducible

from .message_factory_ztf_ms import allwise_match  # same xmatch shape as the ztf factory

LSST_SID = 1  # survey id of LSST in multisurvey
LSST_TID = 1  # telescope id of LSST in multisurvey
BAND_MAP = {"g": 1, "r": 2, "i": 3, "z": 4, "y": 5, "u": 6}  # band letter -> int code, as the lsst parser decodes it


def lsst_epoch(oid, measurement_id, band, mjd, rng):
    """One LSST epoch dict with exactly the keys the lsst parser reads.

    `band` is a letter (u, g, r, i, z, y) and is stored as its int code, the
    type the magstats_ms lsst schema declares.
    """
    return {  # the twelve keys of detection_keys in detections_to_astro_object_lsst
        "oid": oid,  # object id
        "sid": LSST_SID,  # survey id, an int; the parser stringifies it
        "mjd": mjd,  # observation time
        "ra": 150.0 + rng.random() * 1e-4,  # right ascension near a fixed position
        "dec": -30.0 + rng.random() * 1e-4,  # declination near a fixed position
        "psfFlux": rng.uniform(-2000.0, 5000.0),  # difference flux in nJy, may be negative
        "psfFluxErr": rng.uniform(100.0, 300.0),  # difference flux error in nJy
        "scienceFlux": rng.uniform(20000.0, 60000.0),  # science flux in nJy, magnitude ~19.5-20.6
        "scienceFluxErr": rng.uniform(200.0, 800.0),  # science flux error in nJy, mag error < 1
        "tid": LSST_TID,  # telescope id
        "band": BAND_MAP[band],  # int band code
        "measurement_id": measurement_id,  # unique id of the epoch
    }


def generate_message_lsst(
    oid: int = 1152921504606847001,
    bands=("u", "g", "r", "i"),
    n_sources=5,
    n_previous=3,
    n_forced=4,
    seed=7,
    with_xmatch=False,
) -> dict:
    """Build one LSST message with `sources`, `previous_sources` and `forced_sources`.

    sources[1] gets `band: None` and sources[2] gets `scienceFlux: -5.0`, so
    `n_sources` must be at least 3. `n_forced` must be at least 1 so that
    `forced_sources` is non-empty. `with_xmatch=True` attaches an allwise match
    under `xmatches`.
    """
    assert n_sources >= 3, "the two pinned edge rows need at least 3 sources"  # guard the pinned indices
    assert n_forced >= 1, "forced_sources must be non-empty"  # guard the third edge row
    rng = random.Random(seed)  # private generator, seeded per message
    mjd = 60600.0  # first observation time
    mid = 5000  # first measurement id

    sources, previous_sources, forced_sources = [], [], []  # the three epoch arrays
    for i in range(n_sources):  # build the alert's own sources
        sources.append(lsst_epoch(oid, mid, bands[i % len(bands)], mjd, rng))  # cycle through bands
        mjd += 1.3  # advance time
        mid += 1  # next measurement id
    for i in range(n_previous):  # build the previous sources
        previous_sources.append(lsst_epoch(oid, mid, bands[i % len(bands)], mjd, rng))  # cycle through bands
        mjd += 1.3  # advance time
        mid += 1  # next measurement id
    for i in range(n_forced):  # build the forced sources
        forced_sources.append(lsst_epoch(oid, mid, bands[i % len(bands)], mjd, rng))  # cycle through bands
        mjd += 1.3  # advance time
        mid += 1  # next measurement id

    sources[1]["band"] = None  # edge row: no band, pre_execute drops it
    sources[2]["scienceFlux"] = -5.0  # edge row: negative flux, dropped from the magnitude half

    message = {  # the message keys the step reads for lsst
        "oid": oid,  # object id
        "sid": LSST_SID,  # survey id
        "measurement_id": [s["measurement_id"] for s in sources],  # ids of the alert's sources
        "meanra": 150.0,  # mean right ascension
        "meandec": -30.0,  # mean declination
        "sources": sources,  # alert sources
        "previous_sources": previous_sources,  # earlier sources
        "forced_sources": forced_sources,  # forced photometry
    }
    if with_xmatch:  # attach an allwise match only when asked
        message["xmatches"] = allwise_match(oid)  # same shape the xmatch client returns
    return message  # the finished message
