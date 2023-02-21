import gzip
import io
import warnings
from typing import List, Tuple

import pandas as pd
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from atlas_stamp_classifier.inference import AtlasStampClassifier

from .base import BaseStrategy


class ATLASStrategy(BaseStrategy):
    FIELDS = ["FILTER", "AIRMASS", "SEEING", "SUNELONG", "MANGLE"]

    def __init__(self):
        self.model = AtlasStampClassifier()
        super().__init__("atlas_stamp_classifier", "1.0.0")

    @staticmethod
    def _extract_ra_dec(header: dict) -> Tuple[float, float]:
        header["CTYPE1"] += "-SIP"
        header["CTYPE2"] += "-SIP"
        header["RADESYSa"] = header["RADECSYS"]

        del_fields = [
            "CNPIX1",
            "CNPIX2",
            "RADECSYS",
            "RADESYS",
            "RP_SCHIN",
            "CDANAM",
            "CDSKEW",
        ]
        for field in del_fields:
            if field in header.keys():
                del header[field]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            w = WCS(header, relax=True)
        w.sip = None

        pv = []

        for i in [1, 2]:
            for j in range(30):
                pv_name = "PV" + str(i) + "_" + str(j)
                if pv_name in header.keys():
                    pv_val = (i, j, header[pv_name])
                    pv.append(pv_val)

        w.wcs.set_pv(pv)

        w.wcs.ctype = ["RA---TPV", "DEC--TPV"]

        x, y = header["NAXIS1"] * 0.5 + 0.5, header["NAXIS2"] * 0.5 + 0.5
        return w.wcs_pix2world(x, y, 1)

    def extract_image_from_fits(self, stamp_byte, with_metadata=False):
        with gzip.open(io.BytesIO(stamp_byte), "rb") as fh:
            with fits.open(
                io.BytesIO(fh.read()), memmap=False, ignore_missing_simple=True
            ) as hdu:
                im = hdu[0].data
                header = hdu[0].header
        if not with_metadata:
            return im
        metadata = []
        for field in self.FIELDS:
            metadata.append(header[field])

        metadata.extend(self._extract_ra_dec(header))
        return im, metadata

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict_probs(df)

    def _to_dataframe(self, messages: List[dict]) -> pd.DataFrame:
        data, index = [], []
        for msg in messages:
            candid, mjd = msg["candid"], msg["mjd"]
            science, metadata = self.extract_image_from_fits(
                msg["stamps"]["science"], with_metadata=True
            )
            difference = self.extract_image_from_fits(
                msg["stamps"]["difference"], with_metadata=False
            )
            data.append([candid, mjd, science, difference] + metadata)

            index.append(msg["aid"])

        return pd.DataFrame(
            data=data,
            index=index,
            columns=["candid", "mjd", "red", "diff"] + self.FIELDS + ["ra", "dec"],
        )
