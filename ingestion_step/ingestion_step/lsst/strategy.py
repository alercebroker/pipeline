import pandas as pd
from db_plugins.db.sql._connection import PsqlDatabase

from ingestion_step.core.strategy import ParsedData, StrategyInterface
from ingestion_step.core.types import Message
from ingestion_step.core.utils import apply_transforms
from ingestion_step.lsst.database import (
    insert_dia_objects,
    insert_forced_sources,
    insert_mpc_orbit,
    insert_sources,
    insert_ss_sources,
)
from ingestion_step.lsst.extractor import (
    LsstDiaObjectExtractor,
    LsstDiaSourceExtractor,
    LsstForcedSourceExtractor,
    LsstMpcOrbitExtractor,
    LsstPrvSourceExtractor,
    LsstSsSourceExtractor,
)
from ingestion_step.lsst.transforms import (
    get_dia_object_transforms,
    get_forced_source_transforms,
    get_mpc_orbits_transforms,
    get_source_transforms,
    get_ss_source_transforms,
)


class LsstData(ParsedData):
    dia_sources: pd.DataFrame
    ss_sources: pd.DataFrame
    previous_sources: pd.DataFrame
    forced_sources: pd.DataFrame
    dia_object: pd.DataFrame
    mpc_orbits: pd.DataFrame


class LsstStrategy(StrategyInterface[LsstData]):
    @classmethod
    def parse(cls, messages: list[Message]) -> LsstData:
        dia_sources = LsstDiaSourceExtractor.extract(messages)
        ss_sources = LsstSsSourceExtractor.extract(messages)
        previous_sources = LsstPrvSourceExtractor.extract(messages)
        forced_sources = LsstForcedSourceExtractor.extract(messages)
        dia_object = LsstDiaObjectExtractor.extract(messages)
        mpc_orbits = LsstMpcOrbitExtractor.extract(messages)

        source_transforms = get_source_transforms()
        ss_source_transforms = get_ss_source_transforms()
        forced_source_transforms = get_forced_source_transforms()
        dia_object_transforms = get_dia_object_transforms()
        mpc_orbits_transforms = get_mpc_orbits_transforms()

        apply_transforms(dia_sources, source_transforms)
        apply_transforms(ss_sources, ss_source_transforms)
        apply_transforms(previous_sources, source_transforms)
        apply_transforms(forced_sources, forced_source_transforms)
        apply_transforms(dia_object, dia_object_transforms)
        apply_transforms(mpc_orbits, mpc_orbits_transforms)

        return LsstData(
            dia_sources=dia_sources,
            ss_sources=ss_sources,
            previous_sources=previous_sources,
            forced_sources=forced_sources,
            dia_object=dia_object,
            mpc_orbits=mpc_orbits,
        )

    @classmethod
    def insert_into_db(
        cls, driver: PsqlDatabase, parsed_data: LsstData, chunk_size: int | None = None
    ):
        insert_dia_objects(driver, parsed_data["dia_object"], chunk_size=chunk_size)
        insert_mpc_orbit(driver, parsed_data["mpc_orbits"], chunk_size=chunk_size)
        insert_sources(
            driver,
            parsed_data["dia_sources"],
            on_conflict_do_update=True,
            chunk_size=chunk_size,
        )
        insert_sources(driver, parsed_data["previous_sources"], chunk_size=chunk_size)
        insert_ss_sources(driver, parsed_data["ss_sources"], chunk_size=chunk_size)
        insert_forced_sources(
            driver, parsed_data["forced_sources"], chunk_size=chunk_size
        )

    @classmethod
    def serialize(cls, parsed_data: LsstData) -> list[Message]:
        dia_sources_df = parsed_data["dia_sources"]
        ss_sources_df = parsed_data["ss_sources"]
        dia_object_df = parsed_data["dia_object"]
        previous_sources_df = parsed_data["previous_sources"]
        forced_sources_df = parsed_data["forced_sources"]
        dia_sources_records = dia_sources_df.to_dict("records")
        dia_sources_by_msg = {
            record["message_id"]: record for record in dia_sources_records
        }
        dia_object_lookup = {}
        if not dia_object_df.empty:
            dia_object_records = dia_object_df.to_dict("records")
            dia_object_lookup = {
                record["message_id"]: {
                    k: v for k, v in record.items() if k != "message_id"
                }
                for record in dia_object_records
            }
        ss_sources_lookup = {}
        if not ss_sources_df.empty:
            ss_sources_records = ss_sources_df.to_dict("records")
            ss_sources_lookup = {
                record["message_id"]: {
                    k: v for k, v in record.items() if k != "message_id"
                }
                for record in ss_sources_records
            }
        prv_sources_groups = {}
        if not previous_sources_df.empty:
            prv_sources_groups = (
                previous_sources_df.groupby("message_id")
                .apply(lambda x: x.to_dict("records"), include_groups=False)
                .to_dict()
            )
        forced_sources_groups = {}
        if not forced_sources_df.empty:
            forced_sources_groups = (
                forced_sources_df.groupby("message_id")
                .apply(lambda x: x.to_dict("records"), include_groups=False)
                .to_dict()
            )
        messages = []
        for message_id, source in dia_sources_by_msg.items():
            source_clean = {k: v for k, v in source.items() if k != "message_id"}
            messages.append(
                {
                    "oid": source["oid"],
                    "measurement_id": source["measurement_id"],
                    "source": source_clean,
                    "previous_sources": prv_sources_groups.get(message_id, []),
                    "forced_sources": forced_sources_groups.get(message_id, []),
                    "dia_object": dia_object_lookup.get(message_id),
                    # "ss_object":,
                    "ss_source": ss_sources_lookup.get(message_id),
                }
            )
        return messages

    @classmethod
    def get_key(cls) -> str:
        return "oid"
