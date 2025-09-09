import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.strategy import ParsedData, StrategyInterface
from ingestion_step.core.types import Message
from ingestion_step.core.utils import apply_transforms, groupby_messageid
from ingestion_step.ztf import extractor
from ingestion_step.ztf.database import (
    insert_detections,
    insert_forced_photometry,
    insert_non_detections,
    insert_objects,
)
from ingestion_step.ztf.serializer import (
    serialize_detections,
    serialize_non_detections,
)
from ingestion_step.ztf.transforms import (
    CANDIDATES_TRANSFORMS,
    FP_TRANSFORMS,
    PRV_CANDIDATES_TRANSFORMS,
)


class ZtfData(ParsedData):
    objects: pd.DataFrame
    detections: pd.DataFrame
    prv_detections: pd.DataFrame
    non_detections: pd.DataFrame
    forced_photometries: pd.DataFrame


class ZtfStrategy(StrategyInterface[ZtfData]):
    @classmethod
    def parse(cls, messages: list[Message]) -> ZtfData:
        candidates = extractor.ZtfCandidatesExtractor.extract(messages)
        prv_candidates = extractor.ZtfPrvCandidatesExtractor.extract(messages)
        fp_hists = extractor.ZtfFpHistsExtractor.extract(messages)

        apply_transforms(candidates, CANDIDATES_TRANSFORMS)
        apply_transforms(prv_candidates, PRV_CANDIDATES_TRANSFORMS)
        apply_transforms(fp_hists, FP_TRANSFORMS)

        objects = candidates  # Uses a subsets of the fields from candidate
        detections = candidates
        prv_detections = prv_candidates[prv_candidates["candid"].notnull()]
        non_detections = prv_candidates[prv_candidates["candid"].isnull()]
        forced_photometries = fp_hists

        return ZtfData(
            objects=objects,
            detections=detections,
            prv_detections=prv_detections,
            non_detections=non_detections,
            forced_photometries=forced_photometries,
        )

    @classmethod
    def insert_into_db(cls, driver: PsqlDatabase, parsed_data: ZtfData):
        with driver.session() as session:
            insert_objects(session, parsed_data["objects"])
            insert_detections(session, parsed_data["detections"])
            insert_detections(session, parsed_data["prv_detections"])
            insert_non_detections(session, parsed_data["non_detections"])
            insert_forced_photometry(session, parsed_data["forced_photometries"])
            session.commit()

    @classmethod
    def serialize(cls, parsed_data: ZtfData) -> list[Message]:
        objects = parsed_data["objects"]
        detections = serialize_detections(parsed_data["detections"])
        prv_detections = serialize_detections(parsed_data["prv_detections"])
        forced = serialize_detections(parsed_data["forced_photometries"])
        non_detections = serialize_non_detections(parsed_data["non_detections"])

        message_objects = groupby_messageid(objects)
        message_detections = groupby_messageid(detections)
        message_prv_detections = groupby_messageid(prv_detections)
        message_forced = groupby_messageid(forced)
        message_non_detections = groupby_messageid(non_detections)

        messages: list[Message] = []
        for message_id, objects in message_objects.items():
            detections = message_detections.get(message_id, [])
            prv_detections = message_prv_detections.get(message_id, [])
            forced = message_forced.get(message_id, [])
            non_detections = message_non_detections.get(message_id, [])

            assert len(objects) == 1
            obj = objects[0]

            messages.append(
                {
                    "oid": obj["oid"],
                    "measurement_id": obj["measurement_id"],
                    "detections": detections,
                    "prv_detections": prv_detections,
                    "forced_photometries": forced,
                    "non_detections": non_detections,
                }
            )

        return messages

    @classmethod
    def get_key(cls) -> str:
        return "oid"
