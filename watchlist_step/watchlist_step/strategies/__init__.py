__all__ = ["get_strategy"]

from watchlist_step.strategies.base import BaseStrategy


def get_strategy(name: str) -> BaseStrategy:
    if name == "SortingHat":
        from .sorting_hat import SortingHatStrategy

        return SortingHatStrategy()
    else:
        raise ValueError("Invalid strategy requested.")
