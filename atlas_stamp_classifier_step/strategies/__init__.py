from .base import BaseStrategy
from .atlas import AtlasStrategy


__all__ = ["get_strategy", "BaseStrategy", "AtlasStrategy"]


def get_strategy(name: str):
    if name == "AtlasStrategy":
        return AtlasStrategy()
    raise ValueError(f"Unrecognized strategy name {name}")
