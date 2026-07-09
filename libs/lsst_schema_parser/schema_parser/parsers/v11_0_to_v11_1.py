from .parser import BaseParser


class V11_0ToV11_1Parser(BaseParser):
    """
    Parser to convert schema from version 11.0 to version 11.1.

    The only structural change between v11.0 and v11.1 is that four new
    nullable fields were added to diaSource:
      - trailAlgorithm      ["null", "int"]
      - trail_flag          ["null", "boolean"]
      - reliabilityVersion  ["null", "string"]
      - exposureTime        ["null", "float"]

    v11.0 records lack these keys. They are added with a null value when
    absent (and preserved when already present). All other fields are
    passed through unchanged.
    """

    _DIA_SOURCE_NEW_FIELDS = (
        "trailAlgorithm",
        "trail_flag",
        "reliabilityVersion",
        "exposureTime",
    )

    def _add_new_fields(self, record: dict, fields: tuple) -> dict:
        result = dict(record)
        for field in fields:
            result.setdefault(field, None)
        return result

    def get_parse_map(self, source_data: dict) -> dict:
        dia_source = source_data["diaSource"]
        prv_sources = source_data.get("prvDiaSources")

        return {
            "diaSourceId": self.copy_field(source_data, "diaSourceId"),
            "observation_reason": self.copy_field(source_data, "observation_reason"),
            "target_name": self.copy_field(source_data, "target_name"),
            "diaSource": lambda: self._add_new_fields(
                dia_source, self._DIA_SOURCE_NEW_FIELDS
            ),
            "prvDiaSources": lambda: [
                self._add_new_fields(s, self._DIA_SOURCE_NEW_FIELDS)
                for s in prv_sources
            ] if prv_sources else None,
            "prvDiaForcedSources": self.copy_field(source_data, "prvDiaForcedSources"),
            "diaObject": self.copy_field(source_data, "diaObject"),
            "ssSource": self.copy_field(source_data, "ssSource"),
            "mpc_orbits": self.copy_field(source_data, "mpc_orbits"),
            "cutoutDifference": self.copy_field(source_data, "cutoutDifference"),
            "cutoutScience": self.copy_field(source_data, "cutoutScience"),
            "cutoutTemplate": self.copy_field(source_data, "cutoutTemplate"),
        }
