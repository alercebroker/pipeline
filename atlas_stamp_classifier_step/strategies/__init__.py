__all__ = ["get_strategy"]


def get_strategy(name: str):
    if name == "ATLAS":
        from .atlas import ATLASStrategy
        return ATLASStrategy()
    if name == "ZTF":
        from .ztf import ZTFStrategy
        return ZTFStrategy()
    raise ValueError(f"Unrecognized strategy name {name}")
