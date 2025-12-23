#!/usr/bin/env python3
"""
Script to inspect SIMDv6 FITS files and show column names
"""
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import sys


def inspect_fits_file(filepath, file_type):
    """Inspect a FITS file and show its structure, returning data as DataFrame"""
    print(f"\n{'='*70}")
    print(f"{file_type}: {filepath.name}")
    print(f"{'='*70}")
    
    df = None
    try:
        with fits.open(str(filepath)) as hdul:
            hdul.info()
            print()
            
            # Show columns for each HDU with data
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if hasattr(hdu.data, 'columns'):
                        print(f"\nHDU {i} - Columns:")
                        cols = hdu.data.columns.names
                        for col in cols:
                            print(f"  - {col}")
                        
                        # Convert to DataFrame
                        print(f"\nConverting to DataFrame...")
                        table = Table(hdu.data)
                        df = table.to_pandas()
                        print(f"DataFrame shape: {df.shape}")
                        print(f"\nFirst 5 rows:")
                        print(df.head())
                        print(f"\nDataFrame info:")
                        print(df.info())
                        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return df


def main(base_dir, class_name=None):
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Find subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print("No subdirectories found")
        return
    
    # If class_name is provided, find matching directory
    target_subdir = None
    if class_name:
        for subdir in subdirs:
            if class_name.lower() in subdir.name.lower():
                target_subdir = subdir
                break
        if not target_subdir:
            print(f"Class '{class_name}' not found. Available classes:")
            for subdir in sorted(subdirs):
                print(f"  - {subdir.name}")
            return
    else:
        # Use first subdirectory if no class specified
        target_subdir = sorted(subdirs)[0]
    
    print(f"\nInspecting directory: {target_subdir.name}")
    
    # Find files (uppercase .FITS)
    head_files = list(target_subdir.glob("*_HEAD.FITS"))
    phot_files = list(target_subdir.glob("*_PHOT.FITS"))
    
    head_df = None
    phot_df = None
    
    if head_files:
        head_df = inspect_fits_file(head_files[0], "HEAD FILE")
    else:
        print("No HEAD.FITS file found")
    
    if phot_files:
        phot_df = inspect_fits_file(phot_files[0], "PHOT FILE")
    else:
        print("No PHOT.FITS file found")
    
    return head_df, phot_df


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/alerce/Desktop/repos2/SIMDv6"
    class_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Inspecting SIMDv6 directory: {base_dir}")
    if class_name:
        print(f"Looking for class: {class_name}\n")
    else:
        print("No class specified, will use first available\n")
    
    head_df, phot_df = main(base_dir, class_name)
