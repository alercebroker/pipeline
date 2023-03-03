import pandas as pd
import warnings

class DmdtCalculator:
    def do_dmdt(self, df, dt_min=0.5):
        """
        :param nd:  A dataframe with non detections.
        :type nd: :py:class:`pd.DataFrame`

        :param magstats:  A dataframe with magnitude statistics.
        :type magstats: :py:class:`pd.DataFrame`

        :param dt_min:
        :type dt_min: float

        :return: Compute of dmdt of an object
        :rtype: :py:class:`pd.Series`
        """
        response = {}
        df.reset_index(inplace=True)
        magstat_data = df.iloc[0]
        mjd_first = magstat_data.first_mjd
        mask = df.mjd < (mjd_first - dt_min)
        df_masked = df.loc[mask]

        response["close_nondet"] = df_masked.mjd.max() < df.loc[df.mjd < mjd_first, "mjd"].max()
        # is there some non-detection before the first detection
        if mask.sum() > 0:
            magpsf_first = magstat_data.magpsf_first
            sigmapsf_first = magstat_data.sigmapsf_first
            dmdts = dmdt(magpsf_first,
                         sigmapsf_first,
                         df_masked.diffmaglim,
                         mjd_first,
                         df_masked.mjd)
            idxmin = dmdts.dmsigdt.idxmin()
            min_dmdts = dmdts.loc[idxmin]
            min_dmdts = min_dmdts if isinstance(min_dmdts, pd.Series) else min_dmdts.iloc[0]
            min_nd = df.loc[idxmin]
            min_nd = min_nd if isinstance(min_nd, pd.Series) else min_nd.iloc[0]

            response["dmdt_first"] = min_dmdts.dmsigdt
            response["dm_first"] = magpsf_first - min_nd.diffmaglim
            response["sigmadm_first"] = sigmapsf_first - min_nd.diffmaglim
            response["dt_first"] = min_dmdts["dt"]
        else:
            response["dmdt_first"] = np.nan
            response["dm_first"] = np.nan
            response["sigmadm_first"] = np.nan
            response["dt_first"] = np.nan
        return pd.Series(response)


    def compute(self, light_curves: dict, magstats: pd.DataFrame):
        """ Computes the dmdt for the given light curves and magstats

        :light_curves: Dictionary with the detections and non detections
        :magstats: Dataframe containing the previously calculated magstats
        :returns: A DataFrame with the calculated dmdt
        """
        if len(light_curves["non_detections"]) == 0:
            return pd.DataFrame()
        non_detections = light_curves["non_detections"]
        non_detections["objectId"] = non_detections["oid"]
        if "index" in non_detections.columns:
            non_detections.drop(columns=["index"], inplace=True)
            non_detections.reset_index(drop=True, inplace=True)
        magstats["objectId"] = magstats["oid"]

        magstats.set_index(["objectId", "fid"], inplace=True, drop=True)
        non_dets_magstats = non_dets.join(
            magstats, on=["objectId", "fid"], how="inner", rsuffix="_stats"
        )
        responses = []
        for i, g in non_dets_magstats.groupby(["objectId", "fid"]):
            response = do_dmdt(g)
            response["oid"] = i[0]
            response["fid"] = i[1]
            responses.append(response)
        dmdt = pd.DataFrame(responses)
        magstats.reset_index(inplace=True)

        non_detections.drop(columns=["objectId"], inplace=True)
        magstats.drop(columns=["objectId"], inplace=True)
        return dmdt
