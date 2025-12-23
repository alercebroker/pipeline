#!/usr/bin/env python3
"""
Script to read all phot and head.fits files from SIMWV5 folders,
separate light curves, and convert them to LSST message format
Saves data in Parquet format for efficient storage
"""
import os
from pathlib import Path
from astropy.io import fits
import numpy as np
import pandas as pd


def map_band_to_fid(band_name):
    """
    Map SIMWV5 band name to LSST filter ID
    
    Parameters
    ----------
    band_name : str
        Band name (e.g., 'u', 'g', 'r', 'i', 'z', 'y')
    
    Returns
    -------
    fid : int
        Filter ID (0-5 for ugrizy)
    """
    band_map = {
        'u': 6,
        'g': 1,
        'r': 2,
        'i': 3,
        'z': 4,
        'y': 5,
        'Y': 5,
    }
    return band_map.get(str(band_name).lower(), 0)  # Default to '0' band -> None


def convert_lightcurve_to_dataframe(lc_data, head_row, oid):
    """
    Convert a single light curve to DataFrame rows for Parquet storage
    
    Parameters
    ----------
    lc_data : pandas.DataFrame
        Light curve data from phot file
    head_row : pandas.Series
        Metadata from head file for this light curve
    oid : int
        Object ID
        
    Returns
    -------
    sources_df : pandas.DataFrame
        DataFrame with all sources (detections)
    message_meta : dict
        Message-level metadata
    """
    # Get RA/DEC from HEAD file (constant for all detections)
    ra = float(head_row.get('RA', 0))
    dec = float(head_row.get('DEC', 0))
    
    # Calculate minimum flux for this object (for scienceFlux calculation)
    min_flux = np.abs(lc_data['FLUXCAL'].min())
    
    sources_list = []
    measurement_ids = []
    
    for idx, row in lc_data.iterrows():
        # Calculate SNR
        flux = float(row['FLUXCAL'])
        flux_err = float(row['FLUXCALERR'])
        snr = flux / flux_err if flux_err > 0 else 0
        
        # Calculate scienceFlux: psfFlux + min(psfFlux) + 1
        science_flux = flux + min_flux + 1.0
        science_flux_err = flux_err
        
        measurement_id = int(oid + idx)
        measurement_ids.append(measurement_id)
        
        source = {
            "oid": oid,
            "sid": 1,
            "measurement_id": measurement_id,
            "mjd": float(row['MJD']),
            "band": map_band_to_fid(row['BAND'].strip()),
            "diaSourceId": measurement_id,
            "diaObjectId": oid,
            "ra": ra,
            "dec": dec,
            "x": float(row['XPIX']),
            "y": float(row['YPIX']),
            "psfFlux": flux,
            "psfFluxErr": flux_err,
            "scienceFlux": science_flux,
            "scienceFluxErr": science_flux_err,
        }
        sources_list.append(source)
    
    # Create message metadata (non-source data)
    message_meta = {
        "oid": oid,
        "measurement_ids": measurement_ids,  # Store as list for this message
        "timestamp": int(lc_data['MJD'].min() * 86400) if len(lc_data) > 0 else 0,
        "num_sources": len(sources_list),
        "period": float(head_row.get('LCLIB_PARAM(PERIOD_LCLIB)', -999.0))  # Period from HEAD file
    }
    
    return pd.DataFrame(sources_list), message_meta


def process_simwv5_directory(base_dir, output_dir):
    """
    Process SIMWV5 directory and convert all light curves to Parquet format
    
    Parameters
    ----------
    base_dir : str
        Base directory containing SIMWV5 subdirectories
    output_dir : str
        Output directory for Parquet files
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Clases a procesar
    TARGET_CLASSES = ['d-Scuti', 'EB', 'RRL']
    
    # Find all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} subdirectories to process")
    print(f"Processing only classes: {', '.join(TARGET_CLASSES)}")
    
    total_light_curves = 0
    
    BATCH_SIZE = 100000  # Procesar de 100k en 100k para evitar problemas de RAM
    
    for subdir in sorted(subdirs):
        # Filtrar solo las clases objetivo (comparar por prefijo del nombre)
        class_name = subdir.name.split('_')[0]  # Extraer el nombre de la clase antes del primer _
        if class_name not in TARGET_CLASSES:
            print(f"\nSkipping {subdir.name} (class: {class_name}, not in target classes)")
            continue
            
        print(f"\nProcessing directory: {subdir.name}")
        
        # Crear directorio de salida para este subdirectorio
        subdir_output_path = output_path / subdir.name
        subdir_output_path.mkdir(parents=True, exist_ok=True)
        
        # Archivos de salida para este subdirectorio
        subdir_sources_file = subdir_output_path / "sources.parquet"
        subdir_metadata_file = subdir_output_path / "messages_metadata.parquet"
        
        # Check if already processed
        if subdir_sources_file.exists() and subdir_metadata_file.exists():
            print(f"  ⏭️  SKIPPING: Already processed (found .parquet files)")
            continue
        
        # Find files (case insensitive)
        head_files = list(subdir.glob("*HEAD.FITS")) + list(subdir.glob("*head.fits"))
        phot_files = list(subdir.glob("*PHOT.FITS")) + list(subdir.glob("*phot.fits"))
        
        if not head_files:
            print(f"  No head.fits file found in {subdir.name}")
            continue
            
        if not phot_files:
            print(f"  No phot file found in {subdir.name}")
            continue
        
        try:
            # Read head file
            with fits.open(str(head_files[0])) as head_hdul:
                if len(head_hdul) > 1:
                    head_data = head_hdul[1].data
                    head_df = pd.DataFrame({col: head_data[col] for col in head_data.columns.names})
                else:
                    print(f"  No data in head.fits")
                    continue
            
            # Read phot file
            with fits.open(str(phot_files[0])) as phot_hdul:
                if len(phot_hdul) > 1:
                    phot_data = phot_hdul[1].data
                    phot_df = pd.DataFrame({col: phot_data[col] for col in phot_data.columns.names})
                else:
                    print(f"  No data in phot file")
                    continue
            
            num_light_curves = len(head_df)
            print(f"  Found {num_light_curves:,} light curves with {len(phot_df):,} total observations")
            
            # Decidir tamaño de chunk basado en cantidad de curvas
            CHUNK_SIZE = 50000
            num_chunks = (num_light_curves + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
            
            if num_light_curves > CHUNK_SIZE:
                print(f"  📦 Processing in {num_chunks} chunks of {CHUNK_SIZE} light curves each")
            
            # Procesar en chunks - CADA CHUNK se guarda a disco inmediatamente
            # NO acumulamos NADA en memoria global
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * CHUNK_SIZE
                chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_light_curves)
                
                if num_chunks > 1:
                    print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} (LCs {chunk_start+1} to {chunk_end})...")
                
                chunk_sources = []
                chunk_metadata = []
                
                # Procesar solo este chunk
                for i in range(chunk_start, chunk_end):
                    head_row = head_df.iloc[i]
                    
                    # Check for PTROBS_MIN and PTROBS_MAX columns
                    if 'PTROBS_MIN' not in head_row or 'PTROBS_MAX' not in head_row:
                        if i == chunk_start:  # Solo mostrar warning una vez
                            print(f"  Warning: PTROBS_MIN/MAX not found in head file")
                        continue
                    
                    start = int(head_row['PTROBS_MIN'])
                    end = int(head_row['PTROBS_MAX'])
                    
                    # FITS uses 1-based indexing, pandas uses 0-based
                    start_idx = start - 1
                    end_idx = end - 1
                    
                    if start_idx < 0 or end_idx >= len(phot_df) or end_idx < start_idx:
                        continue
                    
                    # Extract light curve data
                    lc_data = phot_df.iloc[start_idx:end_idx+1].copy()
                    
                    # Use SNID from HEAD file as OID
                    oid = int(head_row['SNID'])
                    
                    # Convert to DataFrames
                    sources_df, message_meta = convert_lightcurve_to_dataframe(lc_data, head_row, oid)
                    
                    chunk_sources.append(sources_df)
                    chunk_metadata.append(message_meta)
                    total_light_curves += 1
                
                # Guardar ESTE chunk a disco inmediatamente
                if chunk_sources:
                    combined_chunk = pd.concat(chunk_sources, ignore_index=True)
                    metadata_chunk = pd.DataFrame(chunk_metadata)
                    
                    # Si es el primer chunk, crear archivo. Si no, append
                    if chunk_idx == 0:
                        combined_chunk.to_parquet(subdir_sources_file, compression='snappy', index=False)
                        metadata_chunk.to_parquet(subdir_metadata_file, compression='snappy', index=False)
                    else:
                        # Append mode: leer existente, concatenar, guardar
                        existing_sources = pd.read_parquet(subdir_sources_file)
                        combined_chunk = pd.concat([existing_sources, combined_chunk], ignore_index=True)
                        combined_chunk.to_parquet(subdir_sources_file, compression='snappy', index=False)
                        
                        existing_metadata = pd.read_parquet(subdir_metadata_file)
                        metadata_chunk = pd.concat([existing_metadata, metadata_chunk], ignore_index=True)
                        metadata_chunk.to_parquet(subdir_metadata_file, compression='snappy', index=False)
                    
                    # Limpiar memoria INMEDIATAMENTE
                    del combined_chunk, metadata_chunk, chunk_sources, chunk_metadata
                    import gc
                    gc.collect()
            
            # Mostrar resumen para este subdirectorio
            final_sources = pd.read_parquet(subdir_sources_file)
            final_metadata = pd.read_parquet(subdir_metadata_file)
            print(f"  ✓ Saved to {subdir_output_path}")
            print(f"  ✓ Total light curves: {len(final_metadata):,}")
            print(f"  ✓ Total sources: {len(final_sources):,}")
            del final_sources, final_metadata
            import gc
            gc.collect()
        
        except Exception as e:
            print(f"  Error processing {subdir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Mostrar resumen final
    print(f"\n{'='*70}")
    print(f"✓ Processing complete!")
    print(f"Total light curves processed: {total_light_curves:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")



if __name__ == "__main__":
    import sys
    
    # Default directories
    simwv5_dir = "SIMWv6"
    output_dir = "simulated_parquet_SIMWv6"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        simwv5_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print(f"Reading SIMWv5 data from: {simwv5_dir}")
    print(f"Writing Parquet files to: {output_dir}")
    print()
    
    process_simwv5_directory(simwv5_dir, output_dir)
