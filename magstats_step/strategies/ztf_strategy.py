from magstats_step.strategies.base_strategy import BaseStrategy
from magstats_step.strategies.constants import *
from typing import List
import pandas as pd


class ZTFMagstatsStrategy(BaseStrategy):
    def compute_magstats(self, detections: List[dict], non_detections: List[dict]):
        detections_df = pd.DataFrame(detections)
        non_detections_df = pd.DataFrame(non_detections)
        # TODO: Reemplazar todos los None por np.nan
        magstats = self.apply_mag_stats(detections_df, non_detections_df)
        return magstats

    def apply_mag_stats(self, df, non_detections_df):
        """
        :param df: Dataframe with detections ordered by mjd
        :type df: :py:class:`pd.DataFrame`

        :return: A pandas dataframe with magnitude statistics
        :rtype: :py:class:`pd.DataFrame`
        """

        df = df.sort_values(['mjd'], inplace=False)
        response = {}

        # corrected at the first detection?
        response['corrected'] = df['corrected'].any()

        stellar = self.compute_stellar(df.iloc[0])
        response.update(stellar)
        response["ndet"] = self.compute_ndet(df)
        #resposne["ndubious"] = self.compute_ndubious(df)
        response["nrfid"] = self.compute_nrfid(df)

        # TODO: Is there some default value for the columns if there are no non corrected or corrected mags?
        magnitude_stats_corr =  self.compute_magnitude_statistics(df[df.corrected], corr=True)
        response.update(magnitude_stats_corr)
        magnitude_stats_non_corr =  self.compute_magnitude_statistics(df[~df.corrected], corr=False)
        response.update(magnitude_stats_non_corr)

        # TODO: We're assuming the columns for the magap statistics are also named mjd and mag
        extra_fields_df = pd.DataFrame(list(df.extra_fields))
        magnitude_stats_ap =  self.compute_magnitude_statistics(extra_fields_df, corr=False, magtype='magap')

        time_statistics = self.compute_time_statistics(df[df.corrected])
        response.update(time_statistics)

        response['saturation_rate'] = self.compute_saturation_rate(df)

        # There's no first_mjd without corrected magnitude
        if response['corrected'].any():
            dmdt_dict = self.calculate_dmdt(non_detections_df, response)
            response.update(dmdt_dict)
        return response

    def calculate_dmdt(self, non_detections, magstats, dt_min=0.5):
        """
        TODO: Documment what this function actually does

        :param nd:  A dataframe with non detections.
        :type nd: :py:class:`pd.DataFrame`

        :param magstats: Previously calculated magstats
        :type magstats: :py:class:`dict`

        :param dt_min:
        :type dt_min: float

        :return: Compute of dmdt of an object
        :rtype: :py:class:`dict`
        """
        response = {}
        if non_detections.empty:
            return response
        mjd_first = magstats['first_mjd']
        mask = non_detections.mjd < (mjd_first - dt_min)

        # All non dets more than dt_min away from first detection
        df_masked = non_detections.loc[mask]

        # is there anything in between?
        response["close_nondet"] = df_masked.mjd.max() < non_detections.loc[non_detections.mjd < mjd_first, "mjd"].max()

        if mask.sum() > 0:
            magpsf_first = magstats['mag_first']
            e_mag_first = magstats['e_mag_first']

            # Calculate dm/dt
            dm_sigma = magpsf_first + e_mag_first - df_masked.diffmaglim
            dt = mjd_first - df_masked.mjd
            dmsigdt = (dm_sigma / dt)
            frame = {
                "dm_sigma": dm_sigma,
                "dt": dt,
                "dmsigdt": dmsigdt
            }
            dmdts = pd.DataFrame(frame)

            # Look for non detection with less dmsigdt
            idxmin = dmdts.dmsigdt.idxmin()
            min_dmdts = dmdts.loc[idxmin]

            min_nd = non_detections.loc[idxmin]

            response["dmdt_first"] = min_dmdts.dmsigdt
            response["dm_first"] = magpsf_first - min_nd.diffmaglim
            response["sigmadm_first"] = e_mag_first - min_nd.diffmaglim
            response["dt_first"] = min_dmdts["dt"]
        else:
            response["dmdt_first"] = np.nan
            response["dm_first"] = np.nan
            response["sigmadm_first"] = np.nan
            response["dt_first"] = np.nan

        return response

    def compute_saturation_rate(self, df):
        """ Calculates the proportion of detections that are saturated

        """
        total = len(df)
        if total == 0:
            return np.nan
        satured = (df["mag"] < MAGNITUDE_THRESHOLD).sum()
        return satured/total


    def compute_time_statistics(self, df):
        if df.empty:
            return {}
        time_statistics = {}
        time_statistics["first_mjd"] = df.iloc[0].mjd
        time_statistics["last_mjd"] = df.iloc[-1].mjd
        return time_statistics

    def compute_magnitude_statistics(self, df, corr, magtype='mag'):
        if df.empty:
            return {}
        suffix = "_corr" if corr else ""

        df_min = df.iloc[0]
        df_max = df.iloc[-1]

        # TODO: In the original code, sigma last and mag first are not always calculated
        magnitude_stats = {}
        magnitude_stats[f"{magtype}_mean{suffix}"] = df.mag.mean()
        magnitude_stats[f"{magtype}_median{suffix}"] = df.mag.median()
        magnitude_stats[f"{magtype}_max{suffix}"] = df.mag.max()
        magnitude_stats[f"{magtype}_min{suffix}"] = df.mag.min()
        magnitude_stats[f"{magtype}_sigma{suffix}"] = df.mag.std()
        magnitude_stats[f"{magtype}_first{suffix}"] = df_min.mag
        magnitude_stats[f"e_{magtype}_first{suffix}"] = df_min.e_mag
        magnitude_stats[f"{magtype}_last{suffix}"] = df_max.mag
        return magnitude_stats

    def compute_ndet(self, df):
        ndet = df.shape[0]
        return ndet

    def compute_ndubious(self, df):
        ndubious = df.dubious.sum()
        return ndubious

    def compute_nrfid(self, df):
        rfids = df.rfid.unique().astype(float)
        rfids = rfids[~np.isnan(rfids)]
        return len(rfids)

    def compute_stellar(self, df_min):
        stellar_dict = {}
        # Sets the assignment to np.nan if the field is None
        distnr = df_min['extra_fields']['distnr'] or np.nan
        distpsnr1 = df_min['extra_fields']['distpsnr1'] or np.nan
        sgscore1 = df_min['extra_fields']['sgscore1'] or np.nan
        chinr = df_min['extra_fields']['chinr'] or np.nan
        sharpnr = df_min['extra_fields']['sharpnr'] or np.nan

        stellar_dict["nearZTF"], stellar_dict["nearPS1"], stellar_dict["stellarZTF"], stellar_dict["stellarPS1"] = self.near_stellar(distnr,
                                                                                                                distpsnr1,
                                                                                                                sgscore1,
                                                                                                                chinr,
                                                                                                                sharpnr)
        stellar_dict["stellar"] = self.is_stellar(stellar_dict["nearZTF"], stellar_dict["nearPS1"], stellar_dict["stellarZTF"], stellar_dict["stellarPS1"])

        return stellar_dict

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
        Determine if object is stellar

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
