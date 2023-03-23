from typing import List
from .compose import compose
from ..calculators import *

CALCULATORS_LIST = ["ra", "dec", "mjd", "ndet", "dmdt", "stellar"]

def magstats_intersection(excluded_calculators: List[str]):
    calculators = {
        "ra": calculate_ra,
        "dec": calculate_dec,
        "mjd": calculate_mjd,
        "ndet": calculate_ndet,
        "dmdt": calculate_dmdt,
        "stellar": calculate_stellar
    }

    for calc in excluded_calculators:
        calculators.pop(calc)

    return calculators


def create_magstats_calculator(excluded_calculators: List[str]):
    calculators = magstats_intersection(excluded_calculators)
    return compose(*calculators.values())
