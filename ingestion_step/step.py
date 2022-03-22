from apf.core.step import GenericStep
from apf.producers import KafkaProducer

from ingestion_step.utils.multi_driver.connection import MultiDriverConnection

from .utils.constants import DET_KEYS, OBJ_KEYS, NON_DET_KEYS
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


from typing import Tuple, List

import numpy as np
import pandas as pd
import logging
import sys

sys.path.insert(0, "../../../../")


class IngestionStep(GenericStep):
    """IngestionStep Description

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
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.prv_candidates_processor = Processor(
            ZTFPrvCandidatesStrategy()
        )  # initial strategy (can change)
        self.detections_corrector = Corrector(
            ZTFCorrectionStrategy()
        )  # initial strategy (can change)
        self.producer = producer
        if config.get("PRODUCER_CONFIG", False):
            self.producer = KafkaProducer(config["PRODUCER_CONFIG"])

        self.driver = db_connection or MultiDriverConnection(config["DB_CONFIG"])
        self.driver.connect()

    def get_objects(self, aids: List[str or int], engine="mongo"):
        """

        Parameters
        ----------
        aids
        engine
        Returns
        -------

        """
        filter_by = {"aid": {"$in": aids}}
        objects = self.driver.query("Object", engine=engine).find_all(filter_by=filter_by, paginate=False)
        if len(objects) == 0 or engine == "mongo":
            return pd.DataFrame(objects, columns=OBJ_KEYS)
        return pd.DataFrame(objects)

    def get_detections(self, aids: List[str or int], engine="mongo"):
        """

        Parameters
        ----------
        aids
        engine
        Returns
        -------

        """
        filter_by = {"aid": {"$in": aids}}
        detections = self.driver.query("Detection", engine=engine).find_all(filter_by=filter_by, paginate=False)
        if len(detections) == 0 or engine == "mongo":
            return pd.DataFrame(detections, columns=DET_KEYS)
        return pd.DataFrame(detections)

    def get_non_detections(self, aids: List[str or int], engine="mongo"):
        """

        Parameters
        ----------
        aids
        engine
        Returns
        -------

        """
        filter_by = {"aid": {"$in": aids}}
        non_detections = self.driver.query("NonDetection", engine=engine).find_all(filter_by=filter_by, paginate=False)
        if len(non_detections) == 0 or engine == "mongo":
            return pd.DataFrame(non_detections, columns=NON_DET_KEYS)
        return pd.DataFrame(non_detections)

    def insert_objects(self, objects: pd.DataFrame, engine="mongo") -> None:
        """
        Insert or update records in database. Insert new objects. Update old objects.

        Parameters
        ----------
        objects: Dataframe of astronomical objects.
        engine
        Returns
        -------

        """
        if engine == "mongo":
            objects.drop_duplicates(["aid"], inplace=True)
        else:
            objects.drop_duplicates(["oid"], inplace=True)
        new_objects = objects["new"]
        objects.drop(columns=["new"], inplace=True)

        to_insert = objects[new_objects]
        to_update = objects[~new_objects]
        self.logger.info(
            f"Inserting {len(to_insert)} and updating {len(to_update)} object(s)"
        )
        if len(to_insert) > 0:
            if engine == "mongo":
                to_insert["_id"] = to_insert["aid"].values
            to_insert.replace({np.nan: None}, inplace=True)
            dict_to_insert = to_insert.to_dict("records")
            self.driver.query("Object", engine=engine).bulk_insert(dict_to_insert)
            del to_insert

        if len(to_update) > 0:
            to_update.replace({np.nan: None}, inplace=True)
            dict_to_update = to_update.to_dict("records")
            filters = []
            for obj in dict_to_update:
                if engine == "mongo":
                    filters.append({"_id": obj["aid"]})
                else:
                    filters.append({"_id": obj["oid"]})
            self.driver.query("Object", engine=engine).bulk_update(dict_to_update, filter_by=filters)

    def insert_detections(self, detections: pd.DataFrame, engine="mongo"):
        """

        Parameters
        ----------
        detections
        engine
        Returns
        -------

        """
        self.logger.info(f"Inserting {len(detections)} new detections")
        detections = detections.where(detections.notnull(), None)
        dict_detections = detections.to_dict("records")
        self.driver.query("Detection", engine=engine).bulk_insert(dict_detections)

    def insert_non_detections(self, non_detections: pd.DataFrame, engine="mongo"):
        """

        Parameters
        ----------
        non_detections
        engine
        Returns
        -------

        """
        self.logger.info(f"Inserting {len(non_detections)} new non_detections")
        non_detections.replace({np.nan: None}, inplace=True)
        dict_non_detections = non_detections.to_dict("records")
        self.driver.query("NonDetection", engine=engine).bulk_insert(dict_non_detections)

    @classmethod
    def calculate_stats_coordinates(cls, coordinates, e_coordinates):
        e_coordinates = e_coordinates / 3600
        num_coordinate = np.sum(coordinates / e_coordinates**2)
        den_coordinate = np.sum(1 / e_coordinates**2)
        mean_coordinate = num_coordinate / den_coordinate
        e_coord = np.sqrt(1 / den_coordinate) * 3600
        return mean_coordinate, e_coord

    def compute_meanra(self, ras, e_ras):
        mean_ra, e_ra = self.calculate_stats_coordinates(ras, e_ras)
        if 0.0 <= mean_ra <= 360.0:
            return mean_ra, e_ra
        else:
            raise ValueError(f"Mean ra must be between 0 and 360 (given {mean_ra})")

    def compute_meandec(self, decs, e_decs):
        mean_dec, e_dec = self.calculate_stats_coordinates(decs, e_decs)
        if -90.0 <= mean_dec <= 90.0:
            return mean_dec, e_dec
        else:
            raise ValueError(f"Mean dec must be between -90 and 90 (given {mean_dec})")

    def apply_objs_stats_from_correction(self, df: pd.DataFrame) -> pd.Series:
        response = {}
        df_mjd = df.mjd
        idx_min = df_mjd.values.argmin()
        df_min = df.iloc[idx_min]
        df_ra = df["ra"]
        df_dec = df["dec"]
        df_e_ra = df["e_ra"]
        df_e_dec = df["e_dec"]

        response["meanra"], response["e_ra"] = self.compute_meanra(df_ra, df_e_ra)
        response["meandec"], response["e_dec"] = self.compute_meandec(df_dec, df_e_dec)
        response["firstmjd"] = df_mjd.min()
        response["lastmjd"] = df_mjd.max()
        response["tid"] = df_min.tid
        response["oid"] = list(df["oid"].unique())
        response["ndet"] = len(df)
        return pd.Series(response)

    def preprocess_objects(self, objects: pd.DataFrame, light_curves: dict):
        """

        Parameters
        ----------
        objects
        light_curves

        Returns
        -------

        """
        # Keep existing objects
        aids = objects["aid"].unique()
        detections = light_curves["detections"]
        detections.drop_duplicates(["candid", "aid"], inplace=True, keep="first")
        detections.reset_index(inplace=True, drop=True)
        # New objects referer to: empirical new objects
        # (without detections in the past) and modified objects
        # (I mean existing objects in database)
        new_objects = detections.groupby("aid").apply(
            self.apply_objs_stats_from_correction
        )
        new_objects.reset_index(inplace=True)
        new_objects["new"] = ~new_objects["aid"].isin(aids)
        return new_objects

    def apply_objs_stats_from_correction_psql(self, df: pd.DataFrame) -> pd.Series:
        # add g-r, corrected, reference change, etc
        response = {}
        df_mjd = df.mjd
        idx_min = df_mjd.values.argmin()
        df_min = df.iloc[idx_min]
        df_ra = df["ra"]
        df_dec = df["dec"]
        df_e_ra = df["e_ra"]
        df_e_dec = df["e_dec"]

        response["meanra"], response["e_ra"] = self.compute_meanra(df_ra, df_e_ra)
        response["meandec"], response["e_dec"] = self.compute_meandec(df_dec, df_e_dec)
        response["firstmjd"] = df_mjd.min()
        response["lastmjd"] = df_mjd.max()
        response["tid"] = df_min.tid
        response["ndet"] = len(df)
        return pd.Series(response)

    def preprocess_objects_psql(self, objects: pd.DataFrame, light_curves: dict):
        """

        Parameters
        ----------
        objects
        light_curves

        Returns
        -------

        """
        # Keep existing objects
        oids = objects["oid"].unique()
        detections = light_curves["detections"]
        detections.drop_duplicates(["candid", "oid"], inplace=True, keep="first")
        detections.reset_index(inplace=True, drop=True)
        # New objects referer to: empirical new objects
        # (without detections in the past) and modified objects
        # (I mean existing objects in database)
        new_objects = detections.groupby("oid").apply(
            self.apply_objs_stats_from_correction_psql
        )
        new_objects.reset_index(inplace=True)
        new_objects["new"] = ~new_objects["oid"].isin(oids)
        return new_objects

    def get_lightcurves(self, oids, engine="mongo"):
        """

        Parameters
        ----------
        oids
        engine
        Returns
        -------

        """
        light_curves = {
            "detections": self.get_detections(oids, engine=engine),
            "non_detections": self.get_non_detections(oids, engine=engine),
        }
        self.logger.info(
            f"Light Curves ({len(oids)} objects) of this batch: "
            + f"{len(light_curves['detections'])} detections,"
            + f" {len(light_curves['non_detections'])}"
            + " non_detections in database"
        )
        return light_curves

    def preprocess_lightcurves(
        self, detections: pd.DataFrame, non_detections: pd.DataFrame, engine="mongo"
    ) -> dict:
        """

        Parameters
        ----------
        detections
        non_detections
        engine
        Returns
        -------

        """
        # Assign a label to difference new detections
        detections["new"] = True
        # Get unique oids from new alerts
        if engine == "psql":
            aids = detections["oid"].unique().tolist()
        else:
            aids = detections["aid"].unique().tolist()
        # Retrieve old detections and non_detections from database
        # and put new label to false
        light_curves = self.get_lightcurves(aids, engine=engine)
        light_curves["detections"]["new"] = False
        light_curves["non_detections"]["new"] = False
        old_detections = light_curves["detections"]

        # Remove tuple of [aid, candid] that are new detections and
        # old detections. This is a mask that retrieve
        # existing tuples on db.
        if engine == "mongo":
            unique_keys_detections = ["aid", "candid"]
        else:
            unique_keys_detections = ["oid", "candid"]
        # Checking if already on the database
        index_detections = pd.MultiIndex.from_frame(detections[unique_keys_detections])
        old_index_detections = pd.MultiIndex.from_frame(
            old_detections[unique_keys_detections]
        )
        detections_already_on_db = index_detections.isin(old_index_detections)
        # Apply mask and get only new detections on detections from stream.
        new_detections = detections[~detections_already_on_db]
        # Get all light curve: only detections since beginning of time
        light_curves["detections"] = pd.concat(
            [old_detections, new_detections], ignore_index=True
        )

        non_detections["new"] = True
        old_non_detections = light_curves["non_detections"]
        if len(non_detections):
            # Using round 5 to have 5 decimals of precision to
            # delete duplicates non_detections
            non_detections["round_mjd"] = non_detections["mjd"].round(5)
            old_non_detections["round_mjd"] = old_non_detections["mjd"].round(5)
            # Remove [aid, fid, round_mjd] that are new non_dets
            # and old non_dets.
            if engine == "mongo":
                unique_keys_non_detections = ["aid", "fid", "round_mjd"]
            else:
                unique_keys_non_detections = ["oid", "fid", "round_mjd"]
            # Checking if already on the database
            index_non_detections = pd.MultiIndex.from_frame(
                non_detections[unique_keys_non_detections]
            )
            old_index_non_detections = pd.MultiIndex.from_frame(
                old_non_detections[unique_keys_non_detections]
            )
            non_dets_already_on_db = index_non_detections.isin(old_index_non_detections)
            # Apply mask and get only new non detections on
            # non detections from stream.
            new_non_detections = non_detections[~non_dets_already_on_db]
            # Get all light curve: only detections since beginning of time
            non_detections = pd.concat(
                [old_non_detections, new_non_detections], ignore_index=True
            )
            non_detections.drop(columns=["round_mjd"], inplace=True)
            light_curves["non_detections"] = non_detections
        return light_curves

    def process_prv_candidates(
        self, alerts: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate previous candidates from alerts.

        The input must be a DataFrame created from a list of GenericAlert.

        Parameters
        ----------
        alerts: A pandas DataFrame created from a list of GenericAlerts.

        Returns A Tuple with detections a non_detections from previous candidates
        -------

        """
        # dicto = {
        #     "ZTF": ZTFPrvCandidatesStrategy()
        # }
        data = alerts[["aid", "oid", "tid", "candid", "ra", "dec", "extra_fields"]]
        detections = []
        non_detections = []
        for tid, subset_data in data.groupby("tid"):
            if tid == "ZTF":
                self.prv_candidates_processor.strategy = ZTFPrvCandidatesStrategy()
            elif "ATLAS" in tid:
                self.prv_candidates_processor.strategy = ATLASPrvCandidatesStrategy()
            else:
                raise ValueError(f"Unknown Survey {tid}")
            det, non_det = self.prv_candidates_processor.compute(subset_data)
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

    def produce(
        self,
        alerts: pd.DataFrame,
        objects: pd.DataFrame,
        light_curves: dict,
        key: str = "aid",
    ) -> None:
        """Produce light curves to topic configured on settings.py

        Parameters
        ----------
        alerts
        objects
        light_curves
        key
        Returns
        -------

        """
        if self.producer is None:
            raise Exception("Kafka producer not configured in settings.py")
        # remove unused columns
        light_curves["detections"].drop(columns=["new"], inplace=True)
        light_curves["non_detections"].drop(columns=["new"], inplace=True)

        # sort by ascending mjd
        objects.sort_values("lastmjd", inplace=True, ascending=True)
        self.logger.info(f"Checking {len(objects)} messages (key={key})")
        n_messages = 0
        for index, row in objects.iterrows():
            _key = row[key]
            aid = row["aid"]
            key_alert = alerts[alerts[key] == _key]
            candid = key_alert["candid"].values[-1]  # get the last candid for this key
            mask_detections = light_curves["detections"][key] == _key
            detections = light_curves["detections"].loc[mask_detections]
            detections.replace({np.nan: None}, inplace=True)
            detections = detections.to_dict("records")
            mask_non_detections = light_curves["non_detections"][key] == _key
            non_detections = light_curves["non_detections"].loc[mask_non_detections]
            non_detections = non_detections.to_dict("records")
            output_message = {
                "aid": str(aid),
                "meanra": row["meanra"],
                "meandec": row["meandec"],
                "ndet": row["ndet"],
                "candid": str(candid),
                "detections": detections,
                "non_detections": non_detections,
            }
            self.producer.produce(output_message, key=aid)
            n_messages += 1
        self.logger.info(f"{n_messages} messages produced")

    def execute_psql(self,
                     alerts: pd.DataFrame,
                     detections: pd.DataFrame,
                     non_detections_prv_candidates: pd.DataFrame):
        # Get just ZTF objects
        self.logger.info("Working on PSQL")
        alerts = alerts[alerts["tid"] == "ZTF"]
        detections = detections[detections["tid"] == "ZTF"]
        non_detections_prv_candidates = non_detections_prv_candidates[non_detections_prv_candidates["tid"] == "ZTF"]
        unique_oids = alerts["oid"].unique().tolist()
        self.logger.info("getting lcs")

        light_curves = self.preprocess_lightcurves(detections, non_detections_prv_candidates, engine="psql")
        objects = self.get_objects(unique_oids, engine="psql")
        objects = self.preprocess_objects_psql(objects, light_curves)
        self.insert_objects(objects, engine="psql")
        new_detections = light_curves["detections"]["new"]
        new_detections = light_curves["detections"][new_detections]
        new_detections.loc["step_id_corr"] = self.version
        new_detections.drop(columns=["new"], inplace=True)
        self.insert_detections(new_detections, engine="psql")
        new_non_detections = light_curves["non_detections"]["new"]
        new_non_detections = light_curves["non_detections"][new_non_detections]
        new_non_detections.drop(columns=["new"], inplace=True)
        self.insert_non_detections(new_non_detections)

    def execute_mongo(self,
                      alerts: pd.DataFrame,
                      detections: pd.DataFrame,
                      non_detections_prv_candidates: pd.DataFrame,
                      ):
        # Get unique alerce ids for get objects from database
        unique_aids = alerts["aid"].unique().tolist()
        # Concat new and old detections and non detections.
        light_curves = self.preprocess_lightcurves(
            detections, non_detections_prv_candidates
        )

        # Getting other tables: retrieve existing objects
        # and create new objects
        objects = self.get_objects(unique_aids)
        objects = self.preprocess_objects(objects, light_curves)
        # Insert new objects and update old objects on database
        self.insert_objects(objects)
        # Insert new detections and put step_version
        new_detections = light_curves["detections"]["new"]
        new_detections = light_curves["detections"][new_detections]
        new_detections.loc["step_id_corr"] = self.version
        new_detections.drop(columns=["new"], inplace=True)
        self.insert_detections(new_detections)
        # Insert new now detections
        new_non_detections = light_curves["non_detections"]["new"]
        new_non_detections = light_curves["non_detections"][new_non_detections]
        new_non_detections.drop(columns=["new"], inplace=True)
        self.insert_non_detections(new_non_detections)
        # produce to some topic
        if self.producer:
            self.produce(alerts, objects, light_curves)
        del light_curves["detections"]
        del light_curves["non_detections"]
        del light_curves
        del objects
        del new_detections
        del new_non_detections

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")
        alerts = pd.DataFrame(messages)
        # If is an empiric alert must has stamp
        alerts["has_stamp"] = True
        # Process previous candidates of each alert
        (
            dets_from_prv_candidates,
            non_dets_from_prv_candidates,
        ) = self.process_prv_candidates(alerts)
        # If is an alert from previous candidate hasn't stamps
        # Concat detections from alerts and detections from previous candidates
        if dets_from_prv_candidates.empty:
            detections = alerts.copy()
        else:
            dets_from_prv_candidates["has_stamp"] = False
            detections = pd.concat(
                [alerts, dets_from_prv_candidates], ignore_index=True
            )
        # Remove alerts with the same candid duplicated.
        # It may be the case that some candids are repeated or some
        # detections from prv_candidates share the candid.
        # We use keep='first' for maintain the candid of empiric detections.
        detections.drop_duplicates(
            "candid", inplace=True, keep="first", ignore_index=True
        )
        # Do correction to detections from stream
        detections = self.correct(detections)
        # Insert/update data on mongo
        self.execute_mongo(alerts, detections, non_dets_from_prv_candidates)
        self.execute_psql(alerts, detections, non_dets_from_prv_candidates)

        self.logger.info(f"Clean batch of data\n")
        del alerts
