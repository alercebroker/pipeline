from abc import ABC, abstractmethod
from typing import Dict, List


class InputMessageParsingStrategy(ABC):
    """
    Abstract base class for parsing input messages from different astronomical surveys.

    This strategy pattern allows the correction pipeline to handle messages from different
    surveys (LSST, ZTF, etc.) that have different data structures and field names.
    Each survey implements its own parsing logic while maintaining a consistent interface.

    The key insight is that all surveys provide similar types of astronomical data
    (detections, non-detections, forced photometry) but organize and name them differently.
    This abstraction normalizes those differences.
    """

    @abstractmethod
    def parse_input_messages(self, messages: List[dict]) -> Dict[str, any]:
        """
        Parse raw input messages from a survey into a standardized data structure.

        Args:
            messages (List[dict]): Raw messages from the survey pipeline. Each message
                                 contains detections, metadata, and survey-specific objects.

        Returns:
            Dict[str, any]: Standardized parsed data with keys such as detections, sources, non_detections
            according to the structure of each survey

            It also adds the flag new, crucial for duplicate handling downstream.
            When combined with database data, this distinguishes fresh pipeline
            data from historical database records.
        """
        pass

    @abstractmethod
    def get_input_schema_config(self) -> Dict[str, any]:
        """
        Return pandas schema for this survey's data structures.

        Due to pandas' aggressive type inference, we lose numerical precision when
        creating DataFrames from dictionaries. This method provides explicit dtype
        schemas that preserve the precision needed for astronomical calculations.

        The schemas are applied via core.schemas.schema_applier.apply_schema() which:
        1. Creates pandas Series with explicit dtypes for each field
        2. Builds DataFrames from those Series to avoid type inference
        3. Preserves numerical precision critical for photometric analysis

        Returns:
            Dict[str, any]: Schemas for the current survey

        Example:
            {
                'sources_schema': {
                    'ra': 'float64',           # Right ascension - needs high precision
                    'dec': 'float64',          # Declination - needs high precision
                    'psfFlux': 'float64',      # PSF flux - critical for photometry
                    'psfFluxErr': 'float64',   # Flux error - needed for uncertainties
                    'mjd': 'float64',          # Modified Julian Date - time precision
                    # ... more fields
                },
                'additional_schemas': {
                    'ss_objects': { ... },     # Solar system object fields
                    'dia_objects': { ... }     # Difference image analysis fields
                }
            }
        """
        pass

        @abstractmethod
        def parse_input_messages(self, messages: List[dict]) -> Dict[str, any]:
            """
            Parse raw input messages from a survey into a standardized data structure.

            Returns:
                Dict[str, any]: Standardized structure with:
                    - 'data': Survey-specific data (DataFrames, etc.)
                    - 'counts': Dict with human-readable names and counts for each data type
                    - 'oids': List of unique object IDs
                    - 'measurement_ids': Dict mapping oids to measurement_ids
            """
            pass
