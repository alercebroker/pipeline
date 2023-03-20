from typing import List
from ..calculators import *

CALCULATORS = {
    "dmdt": calculate_dmdt,
    "ra": calculate_ra,
    "dec": calculate_dec
}

def magstats_intersection(included_calculators: List[str]):
    return [CALCULATORS[calc] for calc in included_calculators]
