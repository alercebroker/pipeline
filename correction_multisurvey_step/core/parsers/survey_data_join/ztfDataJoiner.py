from .surveyDataJoiner import SurveyDataJoiner
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

class ZTFDataJoiner(SurveyDataJoiner):
    """ZTF-specific data joining strategy."""
    
    def process_historical_data(self, historical_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Process ZTF historical data."""
        processed = {}
        
        db_sql_detections_df = historical_data.get('detections', pd.DataFrame())
        
        # Separate detections into detections and previous detections 
        if not db_sql_detections_df.empty:
            processed['db_sql_detections_df'] = db_sql_detections_df[
                db_sql_detections_df["parent_candid"].isnull()
            ]
            processed['db_sql_previous_detections_df'] = db_sql_detections_df[
                ~db_sql_detections_df["parent_candid"].isnull()
            ].drop(columns=["rfid", "drb", "drbversion"]) # Remove columns that are from ddbb but not present in previous detections
        else:
            processed['db_sql_detections_df'] = pd.DataFrame()
            processed['db_sql_previous_detections_df'] = pd.DataFrame()
        
        processed['db_sql_forced_photometries_df'] = historical_data.get('forced_photometry', pd.DataFrame())
        processed['db_sql_non_detections_df'] = historical_data.get('non_detections', pd.DataFrame())

        return processed
    
    def combine_data(self, msg_data: Dict[str, pd.DataFrame], 
                    historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Combine ZTF message and historical data."""
        result = {}
    
        result['detections'] = pd.concat([
            msg_data.get('detections_df', pd.DataFrame()),
            historical_data.get('db_sql_detections_df', pd.DataFrame())
        ], ignore_index=True)
        
        result['previous_detections'] = pd.concat([
            msg_data.get('previous_detections_df', pd.DataFrame()),
            historical_data.get('db_sql_previous_detections_df', pd.DataFrame())
        ], ignore_index=True)
        
        result['forced_photometries'] = pd.concat([
            msg_data.get('forced_photometries_df', pd.DataFrame()),
            historical_data.get('db_sql_forced_photometries_df', pd.DataFrame())
        ], ignore_index=True)

        # Remove from previous sources sources that are already in sources (to not end up with duplicates because of alerts that also appear in prvious_sources)
        unique_sources_set = set(zip(result['detections']['measurement_id'], result['detections']['oid']))
        result['previous_detections'] = result['previous_detections'][
            ~result['previous_detections'].apply(lambda row: (row['measurement_id'], row['oid']) in unique_sources_set, axis=1)
        ]

        result['non_detections'] = pd.concat([
            msg_data.get('non_detections_df', pd.DataFrame()),
            historical_data.get('db_sql_non_detections_df', pd.DataFrame())
        ], ignore_index=True)
        
        return result


#! Added MJD column to sort by in order to deduplicate
#TODO check if this logic is correct
    def post_process_data(self, combined_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Post-process ZTF data: sort and deduplicate."""
        result = {}
        
        # For detections previous detections and forced photometries order so new ones are on top, and 
        # drop duplicates based on measurement_id and oid
        for key in ['detections', 'previous_detections', 'forced_photometries']:
            df = combined_data.get(key, pd.DataFrame())
            if not df.empty:
                # Sort by 'new' column (new=True will be on top)
                df = df.sort_values(["new", "mjd"], ascending=[False, False])
                # Drop duplicates based on measurement_id and oid, keeping first (new=True)
                df = df.drop_duplicates(["measurement_id", "oid"], keep="first")
            result[key] = df
        
        # For non_detections, drop duplicates based on oid, band, mjd, and if empty, add an empty DataFrame
        # with expected columns for ZTF
        non_detections_df = combined_data.get('non_detections', pd.DataFrame())
        if not non_detections_df.empty:
            # Drop duplicates based on oid, band, mjd
            non_detections_df = non_detections_df.drop_duplicates(["oid", "band", "mjd"])
        else:
            # If empty, add an empty DataFrame with expected columns
            non_detections_df = pd.DataFrame(columns=[
                "band", "mjd", "oid", "diffmaglim"
            ])
        result['non_detections'] = non_detections_df
        
        logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        logger.info(f"Obtained {len(result['detections'][result['detections']['new']]) if not result['detections'].empty and 'new' in result['detections'].columns else 0} new detections")
        logger.info(f"Obtained {len(result['previous_detections'][result['previous_detections']['new']]) if not result['previous_detections'].empty and 'new' in result['previous_detections'].columns else 0} new previous detections") 
        logger.info(f"Obtained {len(result['forced_photometries'][result['forced_photometries']['new']]) if not result['forced_photometries'].empty and 'new' in result['forced_photometries'].columns else 0} new forced photometries")
        return result
    