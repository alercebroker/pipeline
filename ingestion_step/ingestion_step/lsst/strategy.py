import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.strategy import ParsedData, StrategyInterface
from ingestion_step.core.types import Message
from ingestion_step.core.utils import apply_transforms, groupby_messageid
from ingestion_step.lsst.database import (
    insert_dia_objects,
    insert_forced_sources,
    insert_mpcorb,
    insert_prv_sources,
    # insert_non_detections,
    insert_sources,
    insert_ss_sources,
    # insert_ss_objects,
)
from ingestion_step.lsst.extractor import (
    LsstDiaObjectExtractor,
    LsstDiaSourceExtractor,
    # LsstSsObjectExtractor,
    LsstForcedSourceExtractor,
    LsstMpcorbExtractor,
    # LsstNonDetectionsExtractor,
    LsstPrvSourceExtractor,
    LsstSsSourceExtractor,
)
from ingestion_step.lsst.transforms import (
    get_dia_object_transforms,
    get_forced_source_transforms,
    get_mpcorb_transforms,
    # get_non_detection_transforms,
    get_source_transforms,
    # get_ss_object_transforms,
    get_ss_source_transforms,
)


class LsstData(ParsedData):
    dia_sources: pd.DataFrame
    ss_sources: pd.DataFrame
    previous_sources: pd.DataFrame
    forced_sources: pd.DataFrame
    # non_detections: pd.DataFrame
    dia_object: pd.DataFrame
    # ss_object: pd.DataFrame
    mpcorbs: pd.DataFrame


class LsstStrategy(StrategyInterface[LsstData]):
    @classmethod
    def parse(cls, messages: list[Message]) -> LsstData:
        dia_sources = LsstDiaSourceExtractor.extract(messages)
        ss_sources = LsstSsSourceExtractor.extract(messages)
        previous_sources = LsstPrvSourceExtractor.extract(messages)
        forced_sources = LsstForcedSourceExtractor.extract(messages)
        # non_detections = LsstNonDetectionsExtractor.extract(messages)
        dia_object = LsstDiaObjectExtractor.extract(messages)
        # ss_object = LsstSsObjectExtractor.extract(messages)
        mpcorbs = LsstMpcorbExtractor.extract(messages)

        source_transforms = get_source_transforms()
        ss_source_transforms = get_ss_source_transforms()
        forced_source_transforms = get_forced_source_transforms()
        # non_detection_transforms = get_non_detection_transforms()
        dia_object_transforms = get_dia_object_transforms()
        # ss_object_transforms = get_ss_object_transforms()
        mpcorbs_transforms = get_mpcorb_transforms()

        apply_transforms(dia_sources, source_transforms)
        apply_transforms(ss_sources, ss_source_transforms)
        apply_transforms(previous_sources, source_transforms)
        apply_transforms(forced_sources, forced_source_transforms)
        # apply_transforms(non_detections, non_detection_transforms)
        apply_transforms(dia_object, dia_object_transforms)
        # apply_transforms(ss_object, ss_object_transforms)
        apply_transforms(mpcorbs, mpcorbs_transforms)

        return LsstData(
            dia_sources=dia_sources,
            ss_sources=ss_sources,
            previous_sources=previous_sources,
            forced_sources=forced_sources,
            # non_detections=non_detections,
            dia_object=dia_object,
            # ss_object=ss_object,
            mpcorbs=mpcorbs,
        )

    @classmethod
    def insert_into_db(cls, driver: PsqlDatabase, parsed_data: LsstData):
        with driver.session() as session:
            insert_dia_objects(session, parsed_data["dia_object"])
            insert_mpcorb(session, parsed_data["mpcorbs"])
            insert_sources(session, parsed_data["dia_sources"])
            insert_prv_sources(session, parsed_data["previous_sources"])
            insert_ss_sources(session, parsed_data["ss_sources"])
            insert_forced_sources(session, parsed_data["forced_sources"])
            session.commit()
        # insert_dia_objects(driver, parsed_data["dia_object"])
        # # insert_ss_objects(driver, parsed_data["ss_object"])
        #
        # insert_sources(driver, parsed_data["dia_sources"])
        # insert_sources(driver, parsed_data["previous_sources"])
        # insert_forced_sources(driver, parsed_data["forced_sources"])
        # # insert_non_detections(driver, parsed_data["non_detections"])

    @classmethod
    def serialize(cls, parsed_data: LsstData) -> list[Message]:
        msg_dia_objects = groupby_messageid(parsed_data["dia_object"])
        # msg_ss_objects = groupby_messageid(parsed_data["ss_object"])
        msg_dia_sources = groupby_messageid(parsed_data["dia_sources"])
        msg_prv_sources = groupby_messageid(parsed_data["previous_sources"])
        msg_forced_sources = groupby_messageid(parsed_data["forced_sources"])
        # msg_non_detections = groupby_messageid(parsed_data["non_detections"])

        messages: list[Message] = []
        for message_id, source in msg_dia_sources.items():
            assert len(source) == 1
            source = source[0]

            dia_object = msg_dia_objects.get(message_id, [None])
            assert len(dia_object) == 1
            dia_object = dia_object[0]

            # ss_object = msg_ss_objects.get(message_id, [None])
            # assert len(ss_object) == 1
            # ss_object = ss_object[0]

            prv_sources = msg_prv_sources.get(message_id, [])
            forced_sources = msg_forced_sources.get(message_id, [])
            # non_detections = msg_non_detections.get(message_id, [])

            messages.append(
                {
                    "oid": source["oid"],
                    "measurement_id": source["measurement_id"],
                    "source": source,
                    "previous_sources": prv_sources,
                    "forced_sources": forced_sources,
                    # "non_detections": non_detections,
                    "dia_object": dia_object,
                    # "ss_object": ss_object,
                }
            )

        return messages

    @classmethod
    def get_key(cls) -> str:
        return "oid"
