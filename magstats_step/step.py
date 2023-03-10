from apf.core.step import GenericStep
import pandas as pd
import logging


class MagstatsStep(GenericStep):
    """MagstatsStep Description
    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)
    """

    def __init__(
        self,
        config={},
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)

    def parse_lightcurves(self, alerts: pd.DataFrame) -> dict:
        """Parses the message data and returns a dictionary with detections and
        non detections

        :alerts: Dataframe with alerts
        :returns: A dictionary with detections and non detections.

        """

        object_detections_list = []
        object_non_detections_list = []

        for alerce_id, alerce_object in alerts.iterrows():
            object_detections = pd.DataFrame(alerce_object["detections"])
            object_detections["id"] = alerce_id

            object_non_detections = pd.DataFrame(alerce_object["non_detections"])
            object_non_detections["id"] = alerce_id

            object_detections_list.append(object_detections)
            object_non_detections_list.append(object_non_detections)

        detections = pd.concat(object_detections_list)
        non_detections = pd.concat(object_non_detections_list)

        # Reset all indexes
        alerts.reset_index(inplace=True)
        detections.reset_index(inplace=True)
        non_detections.reset_index(inplace=True)

        # Create a new dataframe with extra fields and remove it from detections
        extra_fields = list(detections["extra_fields"].values)
        extra_fields = pd.DataFrame(extra_fields)
        del detections["extra_fields"]
        # Join detections with extra fields (old format of detections)
        detections = detections.join(extra_fields)

        detections["magpsf"] = detections["mag"]
        detections["sigmapsf"] = detections["e_mag"]

        light_curves = {"detections": detections, "non_detections": non_detections}

        return light_curves

    def recalculate_magstats(
        self, unique_ids: list, light_curves: dict
    ) -> pd.DataFrame:
        """Given a lightcurve, this function recalculates the magstats using
        the data recieved in the alerts

        :unique_ids: Unique alerce ids to query the database
        :light_curves: The lightcurves dictionary with detections and non detections.
        :returns: The new recalculated magstats for the objects.
        """

        new_magstats = self.magstats_calculator.calculate(light_curves)

        """
        # Identify new entries
        old_magstats = get_catalog(unique_ids, "MagStats", self.driver)
        if not old_magstats.empty:
            magstats_index = pd.MultiIndex.from_frame(old_magstats[["oid", "fid"]])
        else:
            magstats_index = pd.Index([])

        new_magstats_index = pd.MultiIndex.from_frame(new_magstats[["oid", "fid"]])
        new_magstats["new"] = ~new_magstats_index.isin(magstats_index)
        """
        new_magstats["new"] = True

        dmdt_calculator = DmdtCalculator()
        dmdt = dmdt_calculator.compute(light_curves, new_magstats)
        if len(dmdt) > 0:
            new_stats = new_magstats.set_index(["oid", "fid"]).join(
                dmdt.set_index(["oid", "fid"])
            )
            new_stats.reset_index(inplace=True)
        else:
            empty_dmdt = [
                "dmdt_first",
                "dm_first",
                "sigmadm_first",
                "dt_first",
            ]
            new_stats = new_magstats.reindex(
                columns=new_magstats.columns.tolist() + empty_dmdt
            )

        new_stats.set_index(["oid", "fid"], inplace=True)
        new_stats.reset_index(inplace=True)

        return new_stats

    def execute(self, messages: list):
        """TODO: Docstring for execute.
        TODO:

        :messages: TODO
        :returns: TODO

        """
        self.logger.info(f"Processing {len(messages)} alerts")
        # Let's assume that alerts have id, detection and non detections keys.
        alerts = pd.DataFrame(messages)

        light_curves = self.parse_lightcurves(alerts)

        # Get unique alerce ids
        unique_ids = alerts["id"].unique().tolist()

        # Reference
        # reference = get_catalog(unique_oids, "Reference", self.driver)
        # PS1
        # ps1 = get_catalog(unique_oids, "Ps1_ztf", self.driver)

        new_stats = self.recalculate_magstats(unique_ids, light_curves)

        # self.magstats_calculator.insert(new_stats, self.driver)

        self.logger.info(f"Clean batch of data\n")
        del alerts
