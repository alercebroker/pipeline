from apf.core.step import GenericStep
from apf.producers import KafkaProducer

from .utils.magstats import do_magstats, insert_magstats, compute_dmdt

from typing import Tuple, List

import numpy as np
import pandas as pd
import logging
import sys


from magstats_step.utils.multi_driver.connection import MultiDriverConnection

from .utils.old_preprocess import (
    get_catalog,
    compute_dmdt,
    insert_magstats)

sys.path.insert(0, "../../../../")
pd.options.mode.chained_assignment = None
logging.getLogger("GP").setLevel(logging.WARNING)
np.seterr(divide="ignore")


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
        consumer=None,
        config=None,
        level=logging.INFO,
        producer=None,
        db_connection=None,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.producer = producer
        if config.get("PRODUCER_CONFIG", False):
            self.producer = KafkaProducer(config["PRODUCER_CONFIG"])

        self.driver = db_connection or MultiDriverConnection(
            config["DB_CONFIG"]
        )
        self.driver.connect()

 def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")
        # Let's assume that alerts have id, detection and non detections keys.
        alerts = pd.DataFrame(messages)

        detections = alerts['detections'].apply(pd.Series)
        non_detections = alerts['non_detections'].apply(pd.Series)

        # Reset all indexes
        alerts.reset_index(inplace=True)
        detections.reset_index(inplace=True)
        non_detections.reset_index(inplace=True)

        # Get unique oids for ZTF
        unique_oids = alerts["oid"].unique().tolist()

        detections["magpsf"] = detections["mag"]
        detections["sigmapsf"] = detections["e_mag"]


        # Reference
        reference = get_catalog(unique_oids, "Reference", self.driver)
        reference = preprocess_reference(reference, detections)
        # PS1
        ps1 = get_catalog(unique_oids, "Ps1_ztf", self.driver)
        ps1 = preprocess_ps1(ps1, detections)

        light_curves = {
                'detections': detections,
                'non_detections' : non_detectionss
                }

        # compute magstats with historic catalogs
        old_magstats = get_catalog(unique_oids, "MagStats", self.driver)
        new_magstats = do_magstats(
            light_curves, old_magstats, ps1, reference, self.version
        )

        dmdt = compute_dmdt(light_curves, new_magstats)
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
        new_stats.loc[magstat_flags.index, "saturation_rate"] = magstat_flags
        new_stats.reset_index(inplace=True)

        insert_magstats(new_stats)

        self.logger.info(f"Clean batch of data\n")
        del alerts
