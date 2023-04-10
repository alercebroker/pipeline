import importlib.metadata
from .core.corrector import Corrector

__all__ = ["Corrector"]
__version__ = importlib.metadata.version(__package__)
