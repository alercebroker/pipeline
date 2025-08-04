import pandas as pd
from typing import Dict, Any
from .surveyDataJoiner import SurveyDataJoiner
import logging

class LSSTDataJoiner(SurveyDataJoiner):
    """LSST-specific data joining strategy."""
    
    def process_historical_data(self, historical_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Process LSST historical data."""
        processed = {}
        
        db_sql_detections_df = historical_data.get('detections', pd.DataFrame())
        
        # Separate detections into sources and previous sources 
        if not db_sql_detections_df.empty:
            processed['db_sql_sources_df'] = db_sql_detections_df[
                db_sql_detections_df["parentDiaSourceId"].isnull()
            ]
            processed['db_sql_previous_sources_df'] = db_sql_detections_df[
                ~db_sql_detections_df["parentDiaSourceId"].isnull()
            ]
        else:
            processed['db_sql_sources_df'] = pd.DataFrame()
            processed['db_sql_previous_sources_df'] = pd.DataFrame()
        
        processed['db_sql_forced_photometries_df'] = historical_data.get('forced_photometry', pd.DataFrame())
        processed['db_sql_non_detections_df'] = historical_data.get('non_detections', pd.DataFrame())
        processed['db_sql_ss_objects_df'] = historical_data.get('ss_objects', pd.DataFrame())
        processed['db_sql_dia_objects_df'] = historical_data.get('dia_objects', pd.DataFrame())
        
        return processed
    
    def combine_data(self, msg_data: Dict[str, pd.DataFrame], 
                    historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Combine LSST message and historical data."""
        result = {}
        

        result['sources'] = pd.concat([
            msg_data.get('sources_df', pd.DataFrame()),
            historical_data.get('db_sql_sources_df', pd.DataFrame())
        ], ignore_index=True)
        
        result['previous_sources'] = pd.concat([
            msg_data.get('previous_sources_df', pd.DataFrame()),
            historical_data.get('db_sql_previous_sources_df', pd.DataFrame())
        ], ignore_index=True)
        
        result['forced_sources'] = pd.concat([
            msg_data.get('forced_sources_df', pd.DataFrame()),
            historical_data.get('db_sql_forced_photometries_df', pd.DataFrame())
        ], ignore_index=True)

        # Remove from previous sources sources that are already in sources (to not end up with duplicates because of alerts that also appear in prvious_sources)
        unique_sources_set = set(zip(result['sources']['measurement_id'], result['sources']['oid']))
        result['previous_sources'] = result['previous_sources'][
            ~result['previous_sources'].apply(lambda row: (row['measurement_id'], row['oid']) in unique_sources_set, axis=1)
        ]

        result['non_detections'] = pd.concat([
            msg_data.get('non_detections_df', pd.DataFrame()),
            historical_data.get('db_sql_non_detections_df', pd.DataFrame())
        ], ignore_index=True)

        # Added a get for ss_objects and dia_objects in case we eventually extract from the database
        # the SS and DIA objects. If not, they will be empty, or can be modified in the future
        ss_objects = pd.concat([
            msg_data.get('ss_objects_df', pd.DataFrame()),
            historical_data.get('db_sql_ss_objects_df', pd.DataFrame())  
        ], ignore_index=True)

        result['ss_object'] = ss_objects.drop_duplicates()
        
        dia_objects = pd.concat([
            msg_data.get('dia_objects_df', pd.DataFrame()),
            historical_data.get('db_sql_dia_objects_df', pd.DataFrame())  
        ], ignore_index=True)

        result['dia_object'] = dia_objects
        
        return result


#! Added MJD column to sort by in order to deduplicate
#TODO check if this logic is correct
    def post_process_data(self, combined_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Post-process LSST data: sort and deduplicate."""
        result = {}
        
        # For sources previous sources and forced photometries order so new ones are on top, and 
        # drop duplicates based on measurement_id and oid
        for key in ['sources', 'previous_sources', 'forced_sources']:
            df = combined_data.get(key, pd.DataFrame())
            if not df.empty:
                # Sort by 'new' column (new=True will be on top)
                df = df.sort_values(["new", "mjd"], ascending=[False, False])
                # Drop duplicates based on measurement_id and oid, keeping first (new=True)
                df = df.drop_duplicates(["measurement_id", "oid"], keep="first")
            result[key] = df
        
        # For non_detections, drop duplicates based on oid, band, mjd, and if empty, add an empty DataFrame
        # with expected columns for LSST
        non_detections_df = combined_data.get('non_detections', pd.DataFrame())
        if not non_detections_df.empty:
            # Drop duplicates based on oid, band, mjd
            non_detections_df = non_detections_df.drop_duplicates(["oid", "band", "mjd"])
        else:
            # If empty, add an empty DataFrame with expected columns
            non_detections_df = pd.DataFrame(columns=[
                "band", "ccdVisitId", "diaNoise", "diaObjectId", 
                "midpointMjdTai", "mjd", "oid", "ssObjectId"
            ])
        result['non_detections'] = non_detections_df
        
        # Since we don't extract from the database the SS and DIA object, we only keep the ones from the message, meaning 
        # it is not necessary to drop duplicates 
        # TODO CHECK THIS LOGIC => do we want to query db objects in the future then combine them?
        result['ss_object'] = combined_data.get('ss_object', pd.DataFrame())
        result['dia_object'] = combined_data.get('dia_object', pd.DataFrame())

        logger = logging.getLogger(f"alerce.{self.__class__.__name__}")

        logger.info(f"Obtained {len(result['sources'][result['sources']['new']])} new dia sources")
        logger.info(f"Obtained {len(result['previous_sources'][result['previous_sources']['new']])} new previous dia sources")
        logger.info(f"Obtained {len(result['forced_sources'][result['forced_sources']['new']])} new forced dia sources")


        return result
    