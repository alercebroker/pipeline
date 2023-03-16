from .atlas import ATLASStrategy
from .ztf import ZTFStrategy
from .base import BaseStrategy


def corrector_factory(detections, tid) -> BaseStrategy:
    if tid.lower().startswith("ztf"):
        return ZTFStrategy(detections)
    elif tid.lower().startswith("atlas"):
        return ATLASStrategy(detections)
    raise ValueError(f"Unrecognized tid: {tid}")
