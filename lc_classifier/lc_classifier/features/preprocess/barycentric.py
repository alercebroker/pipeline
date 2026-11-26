"""Barycentric time correction shared by the survey preprocessors.

Light from a source reaches the Earth earlier or later than it reaches the
Solar System barycentre, by up to ~8.3 minutes depending on where the Earth is
in its orbit. Periodic and timing features need the times referred to the
barycentre, on a uniform time scale, at the middle of each exposure: the
barycentric Julian date in Barycentric Dynamical Time (BJD_TDB), stored here as
an MJD.

The geometry uses the Earth's centre, not the observatory: the observatory's
offset from the geocentre adds at most ~21 ms, and computing it needs the IERS
Earth-rotation tables, which are not available offline.
"""

from typing import Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from astropy.utils import iers

SECONDS_PER_DAY = 86400.0
# Light travel time across one astronomical unit, in days.
AU_LIGHT_TRAVEL_DAYS = (1 * u.au / c).to_value(u.day)
# Start-time tolerance when matching rows that share one exposure (~0.9 s).
# ZTF exposures start at least ~40 s apart, so this never pairs two exposures.
SAME_EXPOSURE_TOLERANCE_DAYS = 1e-5


def mean_source_direction(ra_deg, dec_deg) -> Optional[np.ndarray]:
    """ICRS unit vector toward the mean position of the given coordinates.

    Averages unit vectors rather than angles: a plain mean of RA puts an object
    whose detections straddle RA 0/360 deg at RA ~180 deg.
    Returns None when no coordinate pair is finite.
    """
    ra = np.deg2rad(pd.to_numeric(pd.Series(ra_deg), errors="coerce").to_numpy(float))
    dec = np.deg2rad(pd.to_numeric(pd.Series(dec_deg), errors="coerce").to_numpy(float))
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not valid.any():
        return None
    ra, dec = ra[valid], dec[valid]
    direction = np.array(
        [
            np.mean(np.cos(dec) * np.cos(ra)),
            np.mean(np.cos(dec) * np.sin(ra)),
            np.mean(np.sin(dec)),
        ]
    )
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return None
    return direction / norm


def _exptime_seconds(dataframe: pd.DataFrame) -> np.ndarray:
    if "exptime" not in dataframe.columns:
        return np.full(len(dataframe), np.nan)
    exptime = pd.to_numeric(dataframe["exptime"], errors="coerce").to_numpy(float)
    return np.where(exptime > 0, exptime, np.nan)


def fill_exptime_from_same_exposure(frames: list) -> list:
    """Per-frame exposure times, borrowing a known value from another row of
    the same exposure (same start time) when a row lacks it.

    ZTF previous detections arrive without `exptime`, but the forced photometry
    of the same exposure carries it.
    """
    exptimes = [_exptime_seconds(df) for df in frames]
    known_mjd = np.concatenate(
        [df["mjd"].to_numpy(float)[np.isfinite(e)] for df, e in zip(frames, exptimes)]
    )
    known_exptime = np.concatenate([e[np.isfinite(e)] for e in exptimes])
    if len(known_mjd) == 0:
        return exptimes

    order = np.argsort(known_mjd)
    known_mjd, known_exptime = known_mjd[order], known_exptime[order]
    filled = []
    for df, exptime in zip(frames, exptimes):
        missing = ~np.isfinite(exptime)
        if missing.any():
            mjd = df["mjd"].to_numpy(float)[missing]
            last = len(known_mjd) - 1
            right = np.clip(np.searchsorted(known_mjd, mjd), 0, last)
            left = np.clip(right - 1, 0, last)
            nearest = np.where(
                np.abs(known_mjd[left] - mjd) <= np.abs(known_mjd[right] - mjd),
                left,
                right,
            )
            match = np.abs(known_mjd[nearest] - mjd) <= SAME_EXPOSURE_TOLERANCE_DAYS
            exptime = exptime.copy()
            exptime[np.flatnonzero(missing)[match]] = known_exptime[nearest[match]]
        filled.append(exptime)
    return filled


def to_bjd_tdb(
    mjd: np.ndarray,
    direction: np.ndarray,
    scale: str,
    mid_exposure_offset_days: Optional[np.ndarray] = None,
) -> np.ndarray:
    """BJD_TDB (as MJD) of the given observation times.

    `mjd` is on the time scale `scale` ("utc" for ZTF, "tai" for LSST).
    `mid_exposure_offset_days` moves exposure start times to mid-exposure.
    """
    mjd = np.asarray(mjd, dtype=float)
    if mid_exposure_offset_days is None:
        mid_exposure_offset_days = np.zeros_like(mjd)
    # UTC -> TDB needs only the leap-second table, which ships with astropy.
    # When that bundled table is past its expiry date, astropy would try to
    # download a fresh one on the first UTC conversion of the process; keep it
    # offline. TAI -> TDB needs no table at all.
    with iers.conf.set_temp("auto_download", False):
        times = Time(mjd, mid_exposure_offset_days, format="mjd", scale=scale).tdb
    earth = get_body_barycentric("earth", times).xyz.to_value(u.au)
    roemer_delay_days = np.dot(direction, earth) * AU_LIGHT_TRAVEL_DAYS
    return times.mjd + roemer_delay_days


def barycentric_time_correction(
    astro_object,
    scale: str,
    default_exptime_s: Optional[float] = None,
) -> None:
    """Replace `mjd` in detections, forced photometry and non-detections by
    the mid-exposure BJD_TDB, keeping the original in `mjd_nohelio`.

    With `default_exptime_s` set, `mjd` is taken as the start of the exposure
    and moved by half of each row's `exptime` (or of `default_exptime_s` when
    the row has none). With it unset, `mjd` is already mid-exposure.
    """
    direction = mean_source_direction(
        astro_object.detections["ra"], astro_object.detections["dec"]
    )
    if direction is None:
        return

    frames = [astro_object.detections]
    for frame in (astro_object.forced_photometry, astro_object.non_detections):
        if frame is not None and len(frame) > 0:
            frames.append(frame)

    if default_exptime_s is None:
        offsets = [None] * len(frames)
    else:
        offsets = [
            np.where(np.isfinite(e), e, default_exptime_s) / 2.0 / SECONDS_PER_DAY
            for e in fill_exptime_from_same_exposure(frames)
        ]

    for frame, offset in zip(frames, offsets):
        corrected = to_bjd_tdb(frame["mjd"].to_numpy(float), direction, scale, offset)
        frame["mjd_nohelio"] = frame["mjd"]
        frame["mjd"] = corrected
