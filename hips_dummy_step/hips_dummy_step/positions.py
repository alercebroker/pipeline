"""Read an alert's sky position, by survey."""

import math
from typing import Any

SURVEYS = ("ztf", "lsst")

# Where each survey's alert keeps the position of the detection that
# triggered it: (field holding the detection, field holding the alert id).
_DETECTION_FIELD = {"ztf": "candidate", "lsst": "diaSource"}
_ALERT_ID_FIELD = {"ztf": "candid", "lsst": "diaSourceId"}


def alert_position(message: dict[str, Any], survey: str) -> tuple[Any, Any]:
    """Return the raw (ra, dec) of the alert's detection, in degrees."""
    detection = message[_DETECTION_FIELD[survey]]
    return detection["ra"], detection["dec"]


def alert_id(message: dict[str, Any], survey: str) -> Any:
    """Return the alert's id, for logs."""
    field = _ALERT_ID_FIELD[survey]
    if field in message:
        return message[field]
    return message.get(_DETECTION_FIELD[survey], {}).get(field)


def clean_position(ra: Any, dec: Any) -> tuple[float, float] | None:
    """Return (ra, dec) in the ranges the image service accepts, or None.

    The service rejects a whole request if any position is out of range
    (ra in [0, 360), dec in [-90, 90]), so bad positions are dropped here.
    ra is wrapped into [0, 360); a dec outside [-90, 90] or a missing or
    non-finite value can't be fixed.
    """
    try:
        ra = float(ra)
        dec = float(dec)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(ra) and math.isfinite(dec)):
        return None
    if not -90.0 <= dec <= 90.0:
        return None
    ra = ra % 360.0
    if ra >= 360.0:  # a tiny negative ra wraps to exactly 360.0
        ra = 0.0
    return ra, dec
