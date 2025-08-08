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
    get_dia_object_transforms,
    get_forced_source_transforms,
    get_non_detection_transforms,
    get_source_transforms,
    get_ss_object_transforms,
)

from collections import defaultdict
from typing import Any


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

        source_transforms = get_source_transforms()
        forced_source_transforms = get_forced_source_transforms()
        non_detection_transforms = get_non_detection_transforms()
        dia_object_transforms = get_dia_object_transforms()
        ss_object_transforms = get_ss_object_transforms()

        apply_transforms(sources, source_transforms)
        apply_transforms(previous_sources, source_transforms)
        apply_transforms(forced_sources, forced_source_transforms)
        apply_transforms(non_detections, non_detection_transforms)
        apply_transforms(dia_object, dia_object_transforms)
        apply_transforms(ss_object, ss_object_transforms)
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
        with driver.session() as session:
            insert_dia_objects(session, parsed_data["dia_object"])
            insert_ss_objects(session, parsed_data["ss_object"])
            insert_sources(session, parsed_data["sources"])
            insert_sources(session, parsed_data["previous_sources"])
            insert_forced_sources(session, parsed_data["forced_sources"])
            insert_non_detections(session, parsed_data["non_detections"])
            session.commit()
    

    @staticmethod
    def groupby_messageid(df: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
        """Changes to make the groupby faster using defaultdict"""
        if df.empty:
            return {}
        
        # Get message_ids first
        message_ids = df['message_id'].values
        
        # Convert to records but exclude the message_id column
        df_without_msg_id = df.drop(columns=['message_id'])
        records = df_without_msg_id.to_dict('records')
        
        # Use defaultdict
        result = defaultdict(list)
        for record, msg_id in zip(records, message_ids):
            result[msg_id].append(record)
        
        return dict(result)

    

    @classmethod
    def serialize(cls, parsed_data: LsstData) -> list[Message]:
        # Group all DataFrames in one go using the faster method
        msg_dia_objects = cls.groupby_messageid(parsed_data["dia_object"])
        msg_ss_objects = cls.groupby_messageid(parsed_data["ss_object"])
        msg_sources = cls.groupby_messageid(parsed_data["sources"])
        msg_prv_sources = cls.groupby_messageid(parsed_data["previous_sources"])
        msg_forced_sources = cls.groupby_messageid(parsed_data["forced_sources"])
        msg_non_detections = cls.groupby_messageid(parsed_data["non_detections"])

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

    @classmethod
    def get_key(cls) -> str:
        return "oid"
