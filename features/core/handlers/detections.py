import pandas as pd

from .. import utils
from ._base import BaseAlertHandler


class DetectionsHandler(BaseAlertHandler):
    INDEX = "candid"
    UNIQUE = "candid"
    NON_NULL_COLUMNS = BaseAlertHandler.NON_NULL_COLUMNS + ["mag", "e_mag", "isdiffpos", "ra", "dec"]

    @classmethod
    def _create_df(cls, alerts: list[dict], **kwargs) -> pd.DataFrame:
        extras = kwargs.pop("extras", None)
        df = super()._create_df(alerts, **kwargs)
        cls.__add_ml_magnitudes(df)
        if extras:
            df = df.join(cls.__extract_extra_fields(alerts, extras))
        return df

    @staticmethod
    def __add_ml_magnitudes(df: pd.DataFrame):
        df["mag_ml"] = df["mag"]
        df["e_mag_ml"] = df["e_mag"]
        if "corrected" in df and df["corrected"].any():
            df["mag_ml"][df["corrected"]] = df["mag_corr"]
            df["e_mag_ml"][df["corrected"]] = df["e_mag_corr_ext"]

    @classmethod
    def __extract_extra_fields(cls, alerts: list[dict], extras: list[str]) -> pd.DataFrame:
        df = {alert[cls.INDEX]: alert["extra_fields"] for alert in alerts}
        df = pd.DataFrame.from_dict(df, orient="index", columns=extras)
        return df.reset_index(names=[cls.INDEX]).drop_duplicates(cls.INDEX).set_index(cls.INDEX)

    def get_colors(
        self, func: str, bands: tuple[str, str], *, surveys: str | tuple[str, ...] = (), ml: bool = False
    ) -> pd.Series:
        first, second = bands

        mags = self.get_aggregate(f"mag{'_ml' if ml else ''}", func, by_fid=True, surveys=surveys, bands=bands)
        mags = utils.fill_index(mags, fid=bands)
        return mags.xs(first, level="fid") - mags.xs(second, level="fid")

    def get_count_by_sign(self, sign: int, *, by_fid: bool = False, bands: str | tuple[str, ...] = ()) -> pd.Series:
        counts = self.get_grouped(by_fid=by_fid, bands=bands)["isdiffpos"].value_counts()
        if by_fid and bands:
            return utils.fill_index(counts, fill_value=0, fid=bands, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
        return utils.fill_index(counts, fill_value=0, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
