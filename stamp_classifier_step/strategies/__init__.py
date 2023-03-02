from .base import BaseStrategy

__all__ = ["get_strategy"]


def get_strategy(name: str) -> BaseStrategy:
    """Uses dynamic imports to avoid conflicts for requirements/versions"""
    if name == "ATLAS":
        from .atlas import ATLASStrategy

        return ATLASStrategy()
    if name == "ZTF":
        from .ztf import ZTFStrategy

        return ZTFStrategy()
    raise ValueError(f"Unrecognized strategy name {name}")
