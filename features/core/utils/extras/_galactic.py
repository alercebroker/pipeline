import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


def galactic_coordinates(ra: pd.Series, dec: pd.Series, frame: str = "icrs") -> pd.DataFrame:
    galactic = SkyCoord(ra, dec, frame=frame, unit="deg").galactic
    if np.any(ra.index != dec.index):
        raise ValueError("RA and Dec should have the same indices")
    return pd.DataFrame({"gal_b": galactic.b.degree, "gal_l": galactic.l.degree}, index=ra.index)
