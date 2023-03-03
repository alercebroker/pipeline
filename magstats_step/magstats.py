import pandas as pd
import warnings

from magstats_step.utils.constants import (
    MAGSTATS_TRANSLATE,
    MAGSTATS_UPDATE_KEYS,
)

from magstats_step.utils.multi_driver.connection import MultiDriverConnection

class MagStatsCalculator:
    def near_stellar(self, first_distnr, first_distpsnr1, first_sgscore1, first_chinr, first_sharpnr):
        """
        Get if object is near stellar

        :param first_distnr: Distance to nearest source in reference image PSF-catalog within 30 arcsec [pixels]
        :type first_distnr: :py:class:`float`

        :param first_distpsnr1: Distance of closest source from PS1 catalog; if exists within 30 arcsec [arcsec]
        :type first_distpsnr1: :py:class:`float`

        :param first_sgscore1: Star/Galaxy score of closest source from PS1 catalog 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star
        :type first_sgscore1: :py:class:`float`

        :param first_chinr: DAOPhot chi parameter of nearest source in reference image PSF-catalog within 30 arcsec
        :type first_chinr: :py:class:`float`

        :param first_sharpnr: DAOPhot sharp parameter of nearest source in reference image PSF-catalog within 30 arcsec
        :type first_sharpnr: :py:class:`float`


        :return: if the object is near stellar
        :rtype: tuple
        """

        nearZTF = 0 <= first_distnr < DISTANCE_THRESHOLD
        nearPS1 = 0 <= first_distpsnr1 < DISTANCE_THRESHOLD
        stellarPS1 = first_sgscore1 > SCORE_THRESHOLD
        stellarZTF = first_chinr < CHINR_THRESHOLD and SHARPNR_MIN < first_sharpnr < SHARPNR_MAX
        return nearZTF, nearPS1, stellarPS1, stellarZTF


    def is_stellar(self, nearZTF, nearPS1, stellarPS1, stellarZTF):
        """
        Get if object is stellar

        :param nearZTF:
        :type nearZTF: bool

        :param nearPS1:
        :type nearPS1: bool

        :param stellarPS1:
        :type stellarPS1: bool

        :param stellarZTF:
        :type stellarZTF: bool

        :return: if the object is stellar
        :rtype: bool

        """
        return (nearZTF & nearPS1 & stellarPS1) | (nearZTF & ~nearPS1 & stellarZTF)

    def apply_mag_stats(self, df, distnr=None, distpsnr1=None, sgscore1=None, chinr=None, sharpnr=None, flags=False):
        """
        :param df: A dataframe with corrected detections of a candidate.
        :type df: :py:class:`pd.DataFrame`

        :param distnr: Distance to nearest source in reference image PSF-catalog within 30 arcsec [pixels]
        :type distnr: :py:class:`float`

        :param distpsnr1: Distance of closest source from PS1 catalog; if exists within 30 arcsec [arcsec]
        :type distpsnr1: :py:class:`float`

        :param sgscore1: Star/Galaxy score of closest source from PS1 catalog 0 <= sgscore <= 1 where closer to 1 implies higher likelihood of being a star
        :type sgscore1: :py:class:`float`

        :param chinr: DAOPhot chi parameter of nearest source in reference image PSF-catalog within 30 arcsec
        :type chinr: :py:class:`float`

        :param sharpnr: DAOPhot sharp parameter of nearest source in reference image PSF-catalog within 30 arcsec
        :type sharpnr: :py:class:`float`

        :param flags: If you want compute flags, set it like True
        :type flags: boolean

        :return: A pandas dataframe with magnitude statistics
        :rtype: :py:class:`pd.DataFrame`
        """
        response = {}
        # minimum and maximum candid
        idxmin = df.mjd.values.argmin()
        idxmax = df.mjd.values.argmax()

        df_min = df.iloc[idxmin]
        df_max = df.iloc[idxmax]

        # corrected at the first detection?
        response['corrected'] = df_min["corrected"]

        distnr = df_min.distnr if distnr is None else distnr
        distpsnr1 = df_min.distpsnr1 if distpsnr1 is None else distpsnr1
        sgscore1 = df_min.sgscore1 if sgscore1 is None else sgscore1
        chinr = df_min.chinr if chinr is None else chinr
        sharpnr = df_min.sharpnr if sharpnr is None else sharpnr

        response["nearZTF"], response["nearPS1"], response["stellarZTF"], response["stellarPS1"] = self.near_stellar(distnr,
                                                                                                                distpsnr1,
                                                                                                                sgscore1,
                                                                                                                chinr,
                                                                                                                sharpnr)
        response["stellar"] = self.is_stellar(response["nearZTF"], response["nearPS1"], response["stellarZTF"], response["stellarPS1"])
        # number of detections and dubious detections
        response["ndet"] = df.shape[0]
        response["ndubious"] = df.dubious.sum()

        # reference id
        rfids = df.rfid.unique().astype(np.float)
        rfids = rfids[~np.isnan(rfids)]
        response["nrfid"] = len(rfids)

        # psf magnitude statatistics
        response["magpsf_mean"] = df.magpsf.mean()
        response["magpsf_median"] = df.magpsf.median()
        response["magpsf_max"] = df(detections.columns).magpsf.max()
        response["magpsf_min"] = df.magpsf.min()
        response["sigmapsf"] = df.magpsf.std()
        response["magpsf_first"] = df_min.magpsf
        response["sigmapsf_first"] = df_min.sigmapsf
        response["magpsf_last"] = df_max.magpsf

        # psf corrected magnitude statatistics
        response["magpsf_corr_mean"] = df.magpsf_corr.mean()
        response["magpsf_corr_median"] = df.magpsf_corr.median()
        response["magpsf_corr_max"] = df.magpsf_corr.max()
        response["magpsf_corr_min"] = df.magpsf_corr.min()
        response["sigmapsf_corr"] = df.magpsf_corr.std()
        response["magpsf_corr_first"] = df_min.magpsf_corr
        response["magpsf_corr_last"] = df_max.magpsf_corr

        # corrected psf magnitude statistics
        response["magap_mean"] = df.magap.mean()
        response["magap_median"] = df.magap.median()
        response["magap_max"] = df.magap.max()
        response["magap_min"] = df.magap.min()
        response["sigmap"] = df.magap.std()
        response["magap_first"] = df_min.magap
        response["magap_last"] = df_max.magap

        # time statistics
        response["first_mjd"] = df_min.mjd
        response["last_mjd"] = df_max.mjd

        # flags
        if flags:
            response["saturation_rate"] = get_flag_saturation(df)
        return pd.Series(response)
    def insert(self, magstats: pd.DataFrame, driver: MultiDriverConnection):
        new_magstats = magstats["new"].astype(bool)
        to_insert = magstats[new_magstats]
        to_update = magstats[~new_magstats]

        if len(to_insert) > 0:
            to_insert.replace({np.nan: None}, inplace=True)
            dict_to_insert = to_insert.to_dict("records")
            driver.query("MagStats", engine="psql").bulk_insert(dict_to_insert)

        if len(to_update) > 0:
            to_update.replace({np.nan: None}, inplace=True)
            to_update.rename(
                columns={**MAGSTATS_TRANSLATE},
                inplace=True,
            )
            filter_by = [
                {"_id": x["oid"], "fid": x["fid"]} for i, x in to_update.iterrows()
            ]
            to_update = to_update[["oid", "fid"] + MAGSTATS_UPDATE_KEYS]
            dict_to_update = to_update.to_dict("records")
            driver.query("MagStats", engine="psql").bulk_update(
                dict_to_update, filter_by=filter_by
            )
    def calculate(
        self,
        light_curves: dict,
        version: str,
        ) -> pd.DataFrame:
        """ Applies the magstats calculation to every object separated by filter
        :light_curves: light curves containing detections and non detections
        :version: TODO
        :return: A Dataframe with the calculated magstats
        """

        detections = light_curves["detections"]
        # TODO Why are we ignoring warnings?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_magstats = detections.groupby(["oid", "fid"], sort=False).apply(
                self.apply_mag_stats
            )
        new_magstats.reset_index(inplace=True)
        new_magstats["step_id_corr"] = version
        new_magstats.drop_duplicates(["oid", "fid"], inplace=True)
        return new_magstats
