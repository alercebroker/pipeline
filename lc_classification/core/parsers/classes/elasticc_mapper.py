class ClassMapper:
    _mapping = {
        "Meta/Other": 100,
        "Residual": 200,
        "NotClassified": 300,
        "Static/Other": 1100,
        "Variable/Other": 2100,
        "Non-Recurring/Other": 2210,
        "SN-like/Other": 2221,
        "Ia": 2222,
        "Ib/c": 2223,
        "II": 2224,
        "Iax": 2225,
        "91bg": 2226,
        "Fast/Other": 2231,
        "KN": 2232,
        "M-dwarf Flare": 2233,
        "Dwarf Novae": 2234,
        "uLens": 2235,
        "Long/Other": 2241,
        "SLSN": 2242,
        "TDE": 2243,
        "ILOT": 2244,
        "CART": 2245,
        "PISN": 2246,
        "Recurring/Other": 2310,
        "Periodic/Other": 2321,
        "Cepheid": 2322,
        "RR Lyrae": 2323,
        "Delta Scuti": 2324,
        "EB": 2325,
        "LPV/Mira": 2326,
        "Non-Periodic/Other": 2331,
        "AGN": 2332,
    }

    @classmethod
    def set_mapping(cls, mapping: dict):
        cls._mapping = mapping

    @classmethod
    def has_mapping(cls, name: str):
        return name in cls._mapping

    @classmethod
    def get_class_value(cls, name: str):
        return cls._mapping[name]

    @classmethod
    def get_class_names(cls):
        return cls._mapping.keys()
