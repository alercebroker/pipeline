import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.strategy import ParsedData, StrategyInterface
from ingestion_step.core.types import Message
from ingestion_step.core.utils import apply_transforms, groupby_messageid
from ingestion_step.lsst.database import (
    insert_dia_objects,
    insert_forced_sources,
    insert_non_detections,
    insert_sources,
    insert_ss_objects,
)
from ingestion_step.lsst.extractor import (
    LsstDiaObjectExtractor,
    LsstForcedSourceExtractor,
    LsstNonDetectionsExtractor,
    LsstPrvSourceExtractor,
    LsstSourceExtractor,
    LsstSsObjectExtractor,
)
from ingestion_step.lsst.transforms import (
    DIA_OBJECT_TRANSFORMS,
    FORCED_SOURCE_TRANSFORMS,
    NON_DETECTION_TRANSFORMS,
    SOURCE_TRANSFORMS,
    SS_OBJECT_TRANSFORMS,
)


class LsstData(ParsedData):
    sources: pd.DataFrame
    previous_sources: pd.DataFrame
    forced_sources: pd.DataFrame
    non_detections: pd.DataFrame
    dia_object: pd.DataFrame
    ss_object: pd.DataFrame


class LsstStrategy(StrategyInterface[LsstData]):
    @classmethod
    def parse(cls, messages: list[Message]) -> LsstData:
        sources = LsstSourceExtractor.extract(messages)
        previous_sources = LsstPrvSourceExtractor.extract(messages)
        forced_sources = LsstForcedSourceExtractor.extract(messages)
        non_detections = LsstNonDetectionsExtractor.extract(messages)
        dia_object = LsstDiaObjectExtractor.extract(messages)
        ss_object = LsstSsObjectExtractor.extract(messages)

        apply_transforms(sources, SOURCE_TRANSFORMS)
        apply_transforms(previous_sources, SOURCE_TRANSFORMS)
        apply_transforms(forced_sources, FORCED_SOURCE_TRANSFORMS)
        apply_transforms(non_detections, NON_DETECTION_TRANSFORMS)
        apply_transforms(dia_object, DIA_OBJECT_TRANSFORMS)
        apply_transforms(ss_object, SS_OBJECT_TRANSFORMS)

        return LsstData(
            sources=sources,
            previous_sources=previous_sources,
            forced_sources=forced_sources,
            non_detections=non_detections,
            dia_object=dia_object,
            ss_object=ss_object,
        )

    @classmethod
    def insert_into_db(cls, driver: PsqlDatabase, parsed_data: LsstData):
        insert_dia_objects(driver, parsed_data["dia_object"])
        insert_ss_objects(driver, parsed_data["ss_object"])

        insert_sources(driver, parsed_data["sources"])
        insert_sources(driver, parsed_data["previous_sources"])
        insert_forced_sources(driver, parsed_data["forced_sources"])
        insert_non_detections(driver, parsed_data["non_detections"])

    @classmethod
    def serialize(cls, parsed_data: LsstData) -> list[Message]:
        msg_dia_objects = groupby_messageid(parsed_data["dia_object"])
        msg_ss_objects = groupby_messageid(parsed_data["ss_object"])
        msg_sources = groupby_messageid(parsed_data["sources"])
        msg_prv_sources = groupby_messageid(parsed_data["previous_sources"])
        msg_forced_sources = groupby_messageid(parsed_data["forced_sources"])
        msg_non_detections = groupby_messageid(parsed_data["non_detections"])

        messages: list[Message] = []
        for message_id, source in msg_sources.items():
            assert len(source) == 1
            source = source[0]

            dia_object = msg_dia_objects.get(message_id, [None])
            assert len(dia_object) == 1
            dia_object = dia_object[0]

            ss_object = msg_ss_objects.get(message_id, [None])
            assert len(ss_object) == 1
            ss_object = ss_object[0]

            prv_sources = msg_prv_sources.get(message_id, [])
            forced_sources = msg_forced_sources.get(message_id, [])
            non_detections = msg_non_detections.get(message_id, [])

            messages.append(
                {
                    "oid": source["oid"],
                    "measurement_id": source["measurement_id"],
                    "source": source,
                    "previous_sources": prv_sources,
                    "forced_sources": forced_sources,
                    "non_detections": non_detections,
                    "dia_object": dia_object,
                    "ss_object": ss_object,
                }
            )

        return messages
