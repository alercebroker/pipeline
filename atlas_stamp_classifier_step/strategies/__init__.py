from .base import BaseStrategy
from .atlas import ATLASStrategy
from .ztf import ZTFStrategy


__all__ = ["get_strategy", "BaseStrategy", "ATLASStrategy", "ZTFStrategy"]


def get_strategy(name: str):
    if name == "ATLASStrategy":
        return ATLASStrategy()
    if name == "ZTFStrategy":
        return ZTFStrategy()
    raise ValueError(f"Unrecognized strategy name {name}")
