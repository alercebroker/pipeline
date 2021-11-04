from apf.core.step import GenericStep
from apf.core import get_class
from apf.producers import KafkaProducer

from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from db_plugins.db.mongo.connection import MongoDatabaseCreator

from .utils.prv_candidates.processor import Processor
from .utils.prv_candidates.strategies import (
    ATLASPrvCandidatesStrategy,
    ZTFPrvCandidatesStrategy,
)

from .utils.correction.corrector import Corrector
from .utils.correction.strategies import (
    ATLASCorrectionStrategy,
    ZTFCorrectionStrategy,
)


from survey_parser_plugins import ALeRCEParser

from typing import Tuple, List

import numpy as np
import pandas as pd
import logging
import sys

sys.path.insert(0, "../../../../")


OBJ_KEYS = [
    "aid",
    "tid",
    "oid",
    "lastmjd",
    "firstmjd",
    "meanra",
    "meandec",
    "sigmara",
    "sigmadec",
]
DET_KEYS = [
    "aid",
    "tid",
    "oid",
    "candid",
    "mjd",
    "fid",
    "ra",
    "dec",
    "rb",
    "mag",
    "sigmag",
]
NON_DET_KEYS = ["aid", "oid", "tid", "mjd", "diffmaglim", "fid"]
COR_KEYS = ["magpsf_corr", "sigmapsf_corr", "sigmapsf_corr_ext"]


class GenericSaveStep(GenericStep):
    """GenericSaveStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        consumer=None,
        config=None,
        level=logging.INFO,
        producer=None,
        db_connection=None,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)

        if "CLASS" in config["PRODUCER_CONFIG"]:
            producer_class = get_class(config["PRODUCER_CONFIG"]["CLASS"])
            producer = producer_class(config["PRODUCER_CONFIG"])
        elif "PARAMS" in config["PRODUCER_CONFIG"]:
            producer = KafkaProducer(config["PRODUCER_CONFIG"])

        self.producer = producer
        self.driver = db_connection or new_DBConnection(MongoDatabaseCreator)
        self.driver.connect(config["DB_CONFIG"])
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.parser = ALeRCEParser()
        self.prv_candidates_processor = Processor(ZTFPrvCandidatesStrategy())
        self.detections_corrector = Corrector(ZTFCorrectionStrategy())

    def get_objects(self, oids: List[str or int]):
        """

        Parameters
        ----------
        oids

        Returns
        -------

        """
        filter_by = {"_id": {"$in": oids}}
        objects = self.driver.query().find_all(
            model=Object, filter_by=filter_by, paginate=False
        )
        return pd.DataFrame(objects, columns=OBJ_KEYS)

    def get_detections(self, oids: List[str or int]):
        """

        Parameters
        ----------
        oids

        Returns
        -------

        """
        filter_by = {"aid": {"$in": oids}}
        detections = self.driver.query().find_all(
            model=Detection, filter_by=filter_by, paginate=False
        )
        return pd.DataFrame(detections, columns=DET_KEYS)

    def get_non_detections(self, oids: List[str or int]):
        """

        Parameters
        ----------
        oids

        Returns
        -------

        """
        filter_by = {"aid": {"$in": oids}}
        non_detections = self.driver.query().find_all(
            model=NonDetection, filter_by=filter_by, paginate=False
        )
        return pd.DataFrame(non_detections, columns=NON_DET_KEYS)

    def insert_objects(self, objects: pd.DataFrame):
        """

        Parameters
        ----------
        objects

        Returns
        -------

        """
        objects = objects.copy()
        objects.drop_duplicates(["aid"], inplace=True)
        new_objects = objects["new"]
        objects.drop(columns=["new"], inplace=True)

        to_insert = objects[new_objects]
        to_update = objects[~new_objects]

        if len(to_insert) > 0:
            self.logger.info(f"Inserting {len(to_insert)} new objects")
            to_insert.replace({np.nan: None}, inplace=True)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query().bulk_insert(dict_to_insert, Object)

        if len(to_update) > 0:
            self.logger.info(f"Updating {len(to_update)} objects")
            to_update.replace({np.nan: None}, inplace=True)
            dict_to_update = to_update.to_dict("records")
            print(dict_to_update)
            instances = []
            new_values = []
            filters = []
            for obj in dict_to_update:
                print(obj)
                instances.append(Object(**obj))
                new_values.append(obj)
                filters.append({"_id": obj["aid"]})
            self.driver.query().bulk_update(
                instances, new_values, filter_fields=filters
            )

    def insert_detections(self, detections: pd.DataFrame):
        """

        Parameters
        ----------
        detections

        Returns
        -------

        """
        self.logger.info(f"Inserting {len(detections)} new detections")
        detections = detections.where(detections.notnull(), None)
        dict_detections = detections.to_dict("records")
        self.driver.query().bulk_insert(dict_detections, Detection)

    def insert_non_detections(self, non_detections: pd.DataFrame):
        """

        Parameters
        ----------
        non_detections

        Returns
        -------

        """
        self.logger.info(f"Inserting {len(non_detections)} new non_detections")
        non_detections.replace({np.nan: None}, inplace=True)
        dict_non_detections = non_detections.to_dict("records")
        self.driver.query().bulk_insert(dict_non_detections, NonDetection)
    
    def calculate_stats_coordinates(self, coordinate, sigma_coordinate):
        num_coordinate = np.sum(coordinate/sigma_coordinate)
        den_coordinate = np.sum(1/sigma_coordinate**2)
        mean_coordinate = num_coordinate/den_coordinate

        return mean_coordinate, den_coordinate
    
    
    def apply_objs_stats_from_correction(self, df):
        response = {}
        df_mjd = df.mjd
        idx_min = df_mjd.values.argmin()
        df_min = df.iloc[idx_min]
        df_ra = df.ra
        df_dec = df.dec
        df_sigmara = df.sigmara
        df_sigmadec = df.sigmadec

        response["meanra"], response["sigmara"] = self.calculate_stats_coordinates(df_ra, df_sigmara)
        response["meandec"], response ["sigmadec"] = self.calculate_stats_coordinates(df_dec, df_sigmadec)
        response["firstmjd"] = df_mjd.min()
        response["lastmjd"] = df_mjd.max()
        response["tid"] = df_min.tid
        response["oid"] = df_min.oid
        response["ndet"] = len(df)
        return pd.Series(response)

    def get_last_alert(self, alerts):
        last_alert = alerts["mjd"].values.argmax()
        filtered_alerts = alerts.loc[:, ["aid"]]
        last_alert = filtered_alerts.iloc[last_alert]
        return last_alert

    def preprocess_objects(
        self, objects: pd.DataFrame, light_curves: dict, alerts: pd.DataFrame
    ):
        """

        Parameters
        ----------
        objects
        light_curves
        alerts

        Returns
        -------

        """
        oids = objects.aid.unique()
        apply_last_alert = lambda x: self.get_last_alert(x)
        last_alerts = alerts.groupby("aid", sort=False).apply(apply_last_alert)
        last_alerts.drop(columns=["aid"], inplace=True)

        detections = light_curves["detections"]
        detections_last_alert = detections.join(last_alerts, on="aid")
        detections_last_alert.drop_duplicates(["candid", "aid"], inplace=True)
        detections_last_alert.reset_index(inplace=True)

        new_objects = detections_last_alert.groupby("aid").apply(
            self.apply_objs_stats_from_correction
        )
        new_objects.reset_index(inplace=True)

        new_names = dict(
            [
                (col, col.replace("-", "_"))
                for col in new_objects.columns
                if "-" in col
            ]
        )

        new_objects.rename(columns={**new_names}, inplace=True)
        new_objects["new"] = ~new_objects.aid.isin(oids)

        return new_objects

    def get_lightcurves(self, oids):
        """

        Parameters
        ----------
        oids

        Returns
        -------

        """
        light_curves = {
            "detections": self.get_detections(oids),
            "non_detections": self.get_non_detections(oids),
        }
        self.logger.info(
            f"Light Curves: {len(light_curves['detections'])} detections, {len(light_curves['non_detections'])} non_detections"
        )
        return light_curves

    def preprocess_lightcurves(
        self, detections: pd.DataFrame, non_detections: pd.DataFrame
    ) -> dict:
        """

        Parameters
        ----------
        detections
        non_detections

        Returns
        -------

        """
        # Assign a label to difference new detections
        detections["new"] = True

        # Get unique oids from new alerts
        oids = detections["oid"].unique().tolist()
        # Retrieve old detections and non_detections from database and put new label to false
        light_curves = self.get_lightcurves(oids)
        light_curves["detections"]["new"] = False
        light_curves["non_detections"]["new"] = False

        old_detections = light_curves["detections"]

        # Remove tuple of [oid, candid] that are new detections and old detections. This is a mask that retrieve
        # existing tuples on db.
        unique_keys_detections = ["oid", "candid"]
        detections_already_on_db = (
            detections[unique_keys_detections]
            .isin(old_detections[unique_keys_detections])
            .values
        )
        detections_already_on_db = np.logical_and(
            detections_already_on_db[:, 0], detections_already_on_db[:, 1]
        )

        # Apply mask and get only new detections on detections from stream.
        new_detections = detections[~detections_already_on_db]

        # Get all light curve: only detections since beginning of time
        light_curves["detections"] = pd.concat(
            [old_detections, new_detections], ignore_index=True
        )

        non_detections["new"] = True
        old_non_detections = light_curves["non_detections"]
        if len(non_detections):
            # Using round 5 to have 5 decimals of precision to delete duplicates non_detections
            non_detections["round_mjd"] = non_detections["mjd"].round(5)
            old_non_detections["round_mjd"] = old_non_detections["mjd"].round(
                5
            )
            # Remove [oid, fid, round_mjd] that are new non_dets and old non_dets.
            unique_keys_non_detections = ["oid", "fid", "round_mjd"]
            non_dets_already_on_db = (
                non_detections[unique_keys_non_detections]
                .isin(old_non_detections[unique_keys_non_detections])
                .values
            )
            non_dets_already_on_db = np.logical_and(
                non_dets_already_on_db[:, 0],
                non_dets_already_on_db[:, 1],
                non_dets_already_on_db[:, 2],
            )
            # Apply mask and get only new non detections on non detections from stream.
            new_non_detections = non_detections[~non_dets_already_on_db]
            # Get all light curve: only detections since beginning of time
            non_detections = pd.concat(
                [old_non_detections, new_non_detections], ignore_index=True
            )
            non_detections.drop(columns=["round_mjd"], inplace=True)
            light_curves["non_detections"] = non_detections
        return light_curves

    @classmethod
    def remove_stamps(cls, alerts: pd.DataFrame) -> None:
        """
        Remove a column that contains any survey stamps.

        Parameters
        ----------
        alerts:  A pandas DataFrame created from a list of GenericAlerts.

        Returns None
        -------

        """
        alerts.drop("stamps", axis=1, inplace=True)

    def process_prv_candidates(
        self, alerts: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate previous candidates from alerts.

        The input must be a DataFrame created from a list of GenericAlert.

        Parameters
        ----------
        alerts: A pandas DataFrame created from a list of GenericAlerts.

        Returns A Tuple with detections an non_detections from previous candidates
        -------

        """
        # dicto = {
        #     "ZTF": ZTFPrvCandidatesStrategy()
        # }
        data = alerts[
            ["oid", "tid", "candid", "ra", "dec", "extra_fields"]
        ].copy()
        detections = []
        non_detections = []
        for tid, subset_data in data.groupby("tid"):
            if tid == "ZTF":
                self.prv_candidates_processor.strategy = (
                    ZTFPrvCandidatesStrategy()
                )
            elif "ATLAS" in tid:
                self.prv_candidates_processor.strategy = (
                    ATLASPrvCandidatesStrategy()
                )
            else:
                raise ValueError(f"Unknown Survey {tid}")
            det, non_det = self.prv_candidates_processor.compute(
                subset_data
            )
            detections.append(det)
            non_detections.append(non_det)
        detections = pd.concat(detections, ignore_index=True)
        non_detections = pd.concat(non_detections, ignore_index=True)
        return detections, non_detections

    def correct(self, detections: pd.DataFrame) -> pd.DataFrame:
        """Correct Detections.

        Parameters
        ----------
        detections

        Returns
        -------

        """
        response = []
        for idx, gdf in detections.groupby("tid"):
            if "ZTF" == idx:
                self.detections_corrector.strategy = ZTFCorrectionStrategy()
            elif "ATLAS" in idx:
                self.detections_corrector.strategy = ATLASCorrectionStrategy()
            else:
                raise ValueError(f"Unknown Survey {idx}")
            corrected = self.detections_corrector.compute(gdf)
            response.append(corrected)
        response = pd.concat(response, ignore_index=True)
        return response

    def produce(self, alerts: pd.DataFrame, light_curves: dict) -> None:
        object_ids = alerts["oid"].unique().tolist()
        self.logger.info(f"Checking {len(object_ids)} messages")

        light_curves["detections"].drop(columns=["new"], inplace=True)
        light_curves["non_detections"].drop(columns=["new"], inplace=True)

        n_messages = 0
        for oid in object_ids:
            candid = alerts[alerts["oid"] == oid]["candid"].values[-1]
            aid = alerts[alerts["oid"] == oid]["aid"].values[-1]
            mask_detections = light_curves["detections"]["oid"] == oid
            detections = light_curves["detections"][mask_detections]
            detections.replace({np.nan: None}, inplace=True)
            detections = detections.to_dict("records")

            mask_non_detections = light_curves["detections"]["oid"] == oid
            non_detections = light_curves["non_detections"][
                mask_non_detections
            ]
            non_detections = non_detections.to_dict("records")

            output_message = {
                "aid": str(aid),
                "oid": str(oid),
                "candid": candid,
                "detections": detections,
                "non_detections": non_detections,
            }
            self.producer.produce(output_message, key=oid)
            n_messages += 1
        self.logger.info(f"{n_messages} messages Produced")

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")
        response = self.parser.parse(messages)
        alerts = pd.DataFrame(response)

        # If is an empiric alert must has stamp
        alerts["has_stamp"] = True

        # Process previous candidates of each alert
        (
            dets_from_prv_candidates,
            non_dets_from_prv_candidates,
        ) = self.process_prv_candidates(alerts)

        # If is an alert from previous candidate hasn't stamps
        dets_from_prv_candidates["has_stamp"] = False

        # Concat detections from alerts and detections from previous candidates
        detections = pd.concat(
            [alerts, dets_from_prv_candidates], ignore_index=True
        )
        # Remove alerts with the same candid duplicated. It may be the case that some candid are repeated or some
        # detections from prv_candidates share the candid. We use keep='first' for maintain the candid of empiric
        # detections.
        detections.drop_duplicates(
            "candid", inplace=True, keep="first", ignore_index=True
        )
        # Removing stamps columns
        self.remove_stamps(detections)
        # Do correction to detections from stream
        detections = self.correct(detections)

        # Concat new and old detections and non detections.
        light_curves = self.preprocess_lightcurves(
            detections, non_dets_from_prv_candidates
        )

        # Get unique alerce ids (maybe can be object id from survey) for get objects from database
        unique_aids = alerts["aid"].unique().tolist()

        # Getting other tables
        objects = self.get_objects(unique_aids)
        objects = self.preprocess_objects(objects, light_curves, alerts)
        self.logger.info("Setting objects flags")

        # Insert new objects and update old objects on database
        self.insert_objects(objects)

        # Insert new detections
        new_detections = light_curves["detections"]["new"]
        new_detections = light_curves["detections"][new_detections]
        new_detections.drop(columns=["new"], inplace=True)
        self.insert_detections(new_detections)

        # Insert new now detections
        new_non_detections = light_curves["non_detections"]["new"]
        new_non_detections = light_curves["non_detections"][new_non_detections]
        new_non_detections.drop(columns=["new"], inplace=True)
        self.insert_non_detections(new_non_detections)

        # Finally produce the lightcurves
        # if self.producer:
        #     self.produce(alerts, light_curves)

        del alerts
        del light_curves
        del objects
        del new_detections
        del new_non_detections
