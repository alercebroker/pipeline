from .parser import BaseParser


class V11_1ToV11_2Parser(BaseParser):
    """
    Parser to convert schema from version 11.1 to version 11.2.

    The only structural change between v11.1 and v11.2 is that seven new
    fields were added to diaSource (to support DECam templates):
      - pixelFlags_spike    ["null", "boolean"]  (inserted mid-record)
      - templateMjdMean     ["null", "double"]
      - templateMjdMin      ["null", "double"]
      - templateMjdMax      ["null", "double"]
      - templateInstrument  "string"   (required, no Avro default)
      - templateVersion     "int"      (required, no Avro default)
      - mixedInstruments    "boolean"  (required, no Avro default)

    v11.1 records lack these keys. Nullable ones are added as null; the three
    required ones get the values Rubin documents for "instrument not known /
    template from the same instrument" (every alert before DECam templates).
    Values already present are preserved. All other fields pass through.
    """

    _DIA_SOURCE_NEW_FIELDS = {
        "pixelFlags_spike": None,
        "templateMjdMean": None,
        "templateMjdMin": None,
        "templateMjdMax": None,
        "templateInstrument": "UNKNOWN",
        "templateVersion": 0,
        "mixedInstruments": False,
    }

    def _add_new_fields(self, record: dict, fields: dict) -> dict:
        result = dict(record)
        for field, value in fields.items():
            result.setdefault(field, value)
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
