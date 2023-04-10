import importlib.metadata
from .core.corrector import Corrector

__all__ = ["Corrector"]

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"
