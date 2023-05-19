"""Only for ELAsTiCC"""
import numpy as np
import pandas as pd


def sn_feature_elasticc(df: pd.DataFrame, first_mjd: pd.Series, first_flux: pd.Series) -> pd.Series:
    aid, = df["id"].unique()  # Should never be called with more than one id at the time
    positive_fraction = (df["mag_ml"] > 0).mean()

    non_det_before = df[(df["mjd"] < first_mjd.loc[aid]) & ~df["detected"]]
    non_det_after = df[(df["mjd"] >= first_mjd.loc[aid]) & ~df["detected"]]

    n_non_det_before = non_det_before["mjd"].count()
    n_non_det_after = non_det_after["mjd"].count()

    try:
        last_flux_before = non_det_before["mag_ml"][non_det_before["mjd"].idxmax()]
    except ValueError:  # Occurs for empty non_det_before
        last_flux_before = np.nan
    max_flux_before = non_det_before["mag_ml"].max()
    median_flux_before = non_det_before["mag_ml"].median()

    dflux_first = first_flux.loc[aid] - last_flux_before
    dflux_median_before = first_flux.loc[aid] - median_flux_before

    max_flux_after = non_det_after["mag_ml"].max()
    median_flux_after = non_det_after["mag_ml"].median()

    return pd.Series(
        {
            "positive_fraction": positive_fraction,
            "dflux_first_det_band": dflux_first,
            "dflux_non_det_band": dflux_median_before,
            "last_flux_before_band": last_flux_before,
            "max_flux_before_band": max_flux_before,
            "max_flux_after_band": max_flux_after,
            "median_flux_before_band": median_flux_before,
            "median_flux_after_band": median_flux_after,
            "n_non_det_before_band": n_non_det_before,
            "n_non_det_after_band": n_non_det_after,
        }
    )
