from typing import List
from .compose import compose
from ..calculators import *


def magstats_intersection(excluded_calculators: List[str]):
    calculators = {
        "ra": calculate_ra,
        "dec": calculate_dec,
        "dmdt": calculate_dmdt,
    }

    for calc in excluded_calculators:
        calculators.pop(calc)

    return calculators


def create_magstats_calculator(excluded_calculators: List[str]):
    calculators = magstats_intersection(excluded_calculators)
    return compose(*calculators)
