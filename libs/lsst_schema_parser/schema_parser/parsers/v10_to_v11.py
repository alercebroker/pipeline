from .parser import BaseParser


class V10ToV11Parser(BaseParser):
    """
    Parser to convert schema from version 10 to version 11.

    The only structural change between v10 and v11 is that ten integer
    fields changed from nullable ["null", "int"] to required "int":
      - diaSource: psfNdata, trailNdata, dipoleNdata, bboxSize
      - diaObject: u/g/r/i/z/y_psfFluxNdata

    Null values in those fields are coerced to 0. All other fields are
    passed through unchanged.
    """

    _DIA_SOURCE_INT_FIELDS = ("psfNdata", "trailNdata", "dipoleNdata", "bboxSize")
    _DIA_OBJECT_INT_FIELDS = (
        "u_psfFluxNdata", "g_psfFluxNdata", "r_psfFluxNdata",
        "i_psfFluxNdata", "z_psfFluxNdata", "y_psfFluxNdata",
    )

    def _coerce_int_fields(self, record: dict, fields: tuple) -> dict:
        result = dict(record)
        for field in fields:
            if result.get(field) is None:
                result[field] = 0
        return result

    def get_parse_map(self, source_data: dict) -> dict:
        dia_source = source_data["diaSource"]
        prv_sources = source_data.get("prvDiaSources")
        dia_object = source_data.get("diaObject")

        return {
            "diaSourceId": self.copy_field(source_data, "diaSourceId"),
            "observation_reason": self.copy_field(source_data, "observation_reason"),
            "target_name": self.copy_field(source_data, "target_name"),
            "diaSource": lambda: self._coerce_int_fields(
                dia_source, self._DIA_SOURCE_INT_FIELDS
            ),
            "prvDiaSources": lambda: [
                self._coerce_int_fields(s, self._DIA_SOURCE_INT_FIELDS)
                for s in prv_sources
            ] if prv_sources else None,
            "prvDiaForcedSources": self.copy_field(source_data, "prvDiaForcedSources"),
            "diaObject": lambda: self._coerce_int_fields(
                dia_object, self._DIA_OBJECT_INT_FIELDS
            ) if dia_object else None,
            "ssSource": self.copy_field(source_data, "ssSource"),
            "mpc_orbits": self.copy_field(source_data, "mpc_orbits"),
            "cutoutDifference": self.copy_field(source_data, "cutoutDifference"),
            "cutoutScience": self.copy_field(source_data, "cutoutScience"),
            "cutoutTemplate": self.copy_field(source_data, "cutoutTemplate"),
        }
