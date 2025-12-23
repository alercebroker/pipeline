#!/usr/bin/env python3
"""
Script to consolidate feature DataFrames from dfs_staging folder.
Creates one DataFrame per feature with rows for different OIDs.
"""

import pandas as pd
import os
import glob
import logging
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def consolidate_features():
    """
    Consolidate all DataFrames in dfs_staging by feature.
    Creates one DataFrame per feature with OIDs and timestamps as rows.
    """
    
    # Define paths
    script_dir = Path(__file__).parent
    dfs_staging_dir = script_dir / 'dfs_staging'
    consolidated_dir = script_dir / 'consolidated_features'
    
    # Create consolidated directory
    consolidated_dir.mkdir(exist_ok=True)
    logger.info(f"Consolidated features will be saved to: {consolidated_dir}")
    
    if not dfs_staging_dir.exists():
        logger.error(f"dfs_staging directory not found: {dfs_staging_dir}")
        return
    
    # Find all CSV files in dfs_staging
    csv_files = list(dfs_staging_dir.glob('*.csv'))
    if not csv_files:
        logger.warning(f"No CSV files found in {dfs_staging_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Dictionary to store data for each feature
    # Structure: {feature_name: {oid_timestamp: {band: value}}}
    features_data = defaultdict(lambda: defaultdict(dict))
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            logger.info(f"Processing: {csv_file.name}")
            
            # Extract OID and timestamp from filename (format: features_OID_timestamp.csv)
            filename_parts = csv_file.stem.split('_')
            if len(filename_parts) >= 3:
                oid = filename_parts[1]
                timestamp = '_'.join(filename_parts[2:])  # Handle multi-part timestamps
                oid_timestamp = f"{oid}_{timestamp}"
            else:
                # Fallback to full filename if parsing fails
                oid_timestamp = csv_file.stem
                oid = csv_file.stem
                timestamp = "unknown"
            
            # Read the DataFrame
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"Empty DataFrame in {csv_file.name}")
                continue
            
            # Process each row in the DataFrame
            for _, row in df.iterrows():
                feature_name = row['feature_name']
                band = row['band']
                value = row['value']
                
                # Store the value for this feature, oid_timestamp, and band
                features_data[feature_name][oid_timestamp][band] = value
                
        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Create consolidated DataFrames for each feature
    logger.info(f"Creating consolidated DataFrames for {len(features_data)} features")
    
    for feature_name, oid_timestamp_data in features_data.items():
        try:
            # Create DataFrame for this feature
            consolidated_data = []
            
            for oid_timestamp, band_data in oid_timestamp_data.items():
                # Split oid_timestamp back to separate columns
                if '_' in oid_timestamp:
                    # Try to find the split point (OID should be all digits)
                    parts = oid_timestamp.split('_')
                    oid = parts[0]
                    timestamp = '_'.join(parts[1:])
                else:
                    oid = oid_timestamp
                    timestamp = "unknown"
                
                # Create a row for this OID and timestamp
                row = {
                    'oid': oid,
                    'timestamp': timestamp,
                    'oid_timestamp': oid_timestamp
                }
                
                # Add columns for each band
                for band, value in band_data.items():
                    row[f'band_{band}'] = value
                
                consolidated_data.append(row)
            
            # Create DataFrame
            feature_df = pd.DataFrame(consolidated_data)
            
            # Sort by OID and then by timestamp for consistency
            feature_df = feature_df.sort_values(['oid', 'timestamp']).reset_index(drop=True)
            
            # Save the consolidated DataFrame
            safe_feature_name = feature_name.replace('/', '_').replace('\\', '_')
            output_file = consolidated_dir / f'{safe_feature_name}.csv'
            feature_df.to_csv(output_file, index=False)
            
            unique_oids = feature_df['oid'].nunique()
            total_records = len(feature_df)
            num_bands = len([col for col in feature_df.columns if col.startswith('band_')])
            
            logger.info(f"Saved {feature_name}: {total_records} records ({unique_oids} unique OIDs), {num_bands} bands -> {output_file.name}")
            
            # Print preview of the DataFrame
            print(f"\n=== {feature_name} ===")
            print(f"Total records: {total_records}")
            print(f"Unique OIDs: {unique_oids}")
            print(f"Records per OID: {total_records/unique_oids:.1f} avg")
            print(feature_df.head())
            
        except Exception as e:
            logger.error(f"Error creating DataFrame for feature {feature_name}: {e}")
    
    logger.info("Consolidation complete!")

def get_feature_summary():
    """
    Print a summary of all consolidated features.
    """
    script_dir = Path(__file__).parent
    consolidated_dir = script_dir / 'consolidated_features'
    
    if not consolidated_dir.exists():
        print("No consolidated_features directory found. Run consolidate_features() first.")
        return
    
    csv_files = list(consolidated_dir.glob('*.csv'))
    if not csv_files:
        print("No consolidated feature files found.")
        return
    
    print("\n=== FEATURE SUMMARY ===")
    print(f"Total features: {len(csv_files)}")
    print("-" * 50)
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            feature_name = csv_file.stem
            num_oids = len(df)
            num_bands = len([col for col in df.columns if col.startswith('band_')])
            
            print(f"{feature_name:<30} | {num_oids:>4} OIDs | {num_bands:>2} bands")
            
        except Exception as e:
            print(f"{csv_file.name:<30} | ERROR: {e}")

if __name__ == "__main__":
    print("=== Feature Consolidation Script ===")
    print("1. Consolidating features from dfs_staging...")
    consolidate_features()
    
    print("\n2. Generating summary...")
    get_feature_summary()