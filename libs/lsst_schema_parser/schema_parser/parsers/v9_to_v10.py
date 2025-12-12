from .parser import BaseParser

class V9ToV10Parser(BaseParser):
    """
    Parser to convert schema from version 9 to version 10.
    """
    
    def get_parse_map(self):
        return {
            "diaSourceId": self.copy_field(self.source_data, "diaSourceId"),
            "observation_reason": self.copy_field(self.source_data, "observation_reason"),
            "target_name": self.copy_field(self.source_data, "target_name"),
            "diaSource": self.copy_field(self.source_data, "diaSource"),
            "prvDiaSources": self.copy_field(self.source_data, "prvDiaSources"),
            "prvDiaForcedSources": self.copy_field(self.source_data, "prvDiaForcedSources"),
            "diaObject": self.copy_field(self.source_data, "diaObject"),
            "ssSource": None,
            "mpc_orbits": None,
            "cutoutDifference": self.copy_field(self.source_data, "cutoutDifference"),
            "cutoutScience": self.copy_field(self.source_data, "cutoutScience"),
            "cutoutTemplate": self.copy_field(self.source_data, "cutoutTemplate"),
        }