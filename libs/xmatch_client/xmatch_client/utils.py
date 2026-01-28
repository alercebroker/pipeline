from collections.abc import Iterable
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], size: int):
    """
    Splits an iterable into batches of a given size.

    Args:
        iterable: The iterable to split
        size: The size of each batch

    Yields:
        iterable: A batch of the iterable
    """
    if size < 1:
        raise ValueError("size must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, size)):
        yield batch
