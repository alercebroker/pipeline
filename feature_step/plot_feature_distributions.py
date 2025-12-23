#!/usr/bin/env python3
"""
Script to plot feature distributions from consolidated features.
Creates distribution plots for each feature-band combination.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# LSST band mapping - translate from numeric codes to letters
LSST_BAND_MAPPING = {"g": 1, "r": 2, "i": 3, "z": 4, "y": 5, "u": 6}
# Reverse mapping for translation from numbers to letters
BAND_CODE_TO_LETTER = {v: k for k, v in LSST_BAND_MAPPING.items()}

def translate_band_name(band_col):
    """
    Translate band column name to use letter instead of number.
    Special case: band_0 returns empty string (no band info in title)
    
    Args:
        band_col (str): Column name like 'band_1', 'band_2', etc.
        
    Returns:
        str: Translated name like 'band_g', 'band_r', etc., or '' for band_0
    """
    if band_col.startswith('band_') and len(band_col.split('_')) == 2:
        try:
            band_number = int(band_col.split('_')[1])
            if band_number == 0:
                return ""  # Special case for band_0 - no band info in title
            elif band_number in BAND_CODE_TO_LETTER:
                return f"band_{BAND_CODE_TO_LETTER[band_number]}"
        except ValueError:
            pass
    return band_col

def plot_feature_distributions():
    """
    Plot distribution of each feature by band.
    Ignores NaN values and skips columns that are completely NaN.
    """
    
    # Define paths
    script_dir = Path(__file__).parent
    consolidated_dir = script_dir / 'consolidated_features'
    plots_dir = script_dir / 'feature_plots'
    
    # Create plots directory
    plots_dir.mkdir(exist_ok=True)
    logger.info(f"Plots will be saved to: {plots_dir}")
    
    if not consolidated_dir.exists():
        logger.error(f"consolidated_features directory not found: {consolidated_dir}")
        logger.info("Please run consolidate_features.py first")
        return
    
    # Find all CSV files in consolidated_features
    csv_files = list(consolidated_dir.glob('*.csv'))
    if not csv_files:
        logger.warning(f"No CSV files found in {consolidated_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} feature files to plot")
    
    total_plots = 0
    skipped_plots = 0
    
    for csv_file in csv_files:
        try:
            feature_name = csv_file.stem
            logger.info(f"Processing feature: {feature_name}")
            
            # Read the DataFrame
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {feature_name}")
                continue
            
            # Get band columns (exclude 'oid' column)
            band_columns = [col for col in df.columns if col.startswith('band_')]
            
            if not band_columns:
                logger.warning(f"No band columns found for {feature_name}")
                continue
            
            # Filter out completely NaN columns
            valid_columns = []
            for col in band_columns:
                if not df[col].isna().all():
                    valid_columns.append(col)
                else:
                    logger.info(f"Skipping {col} for {feature_name} - all NaN values")
            
            if not valid_columns:
                logger.warning(f"All band columns are NaN for {feature_name}")
                skipped_plots += 1
                continue
            
            # Create subplots
            n_bands = len(valid_columns)
            if n_bands == 1:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                axes = [ax]
            else:
                # Calculate grid dimensions
                n_cols = min(3, n_bands)  # Max 3 columns
                n_rows = (n_bands + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
                if n_bands > 1:
                    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            # Plot each band
            for i, band_col in enumerate(valid_columns):
                ax = axes[i] if n_bands > 1 else axes[0]
                
                # Translate band name for display
                display_band_name = translate_band_name(band_col)
                
                # Get data and remove NaN values
                data = df[band_col].dropna()
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{display_band_name} (No Data)')
                    continue
                
                # Check if all values are the same
                if data.nunique() == 1:
                    # Bar plot for single value
                    ax.bar([str(data.iloc[0])], [len(data)])
                    ax.set_title(f'{display_band_name} (n={len(data)})')
                    ax.set_ylabel('Count')
                else:
                    # Histogram for multiple values
                    ax.hist(data, bins=min(30, len(data.unique())), alpha=0.7, edgecolor='black')
                    ax.set_title(f'{display_band_name} (n={len(data)})')
                    ax.set_ylabel('Frequency')
                
                # Add statistics text
                stats_text = f'Mean: {data.mean():.3f}\nStd: {data.std():.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Hide unused subplots
            if n_bands > 1:
                for j in range(len(valid_columns), len(axes)):
                    axes[j].set_visible(False)
            
            # Set main title
            plt.suptitle(f'Distribution of {feature_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            safe_feature_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            plot_filename = plots_dir / f'{safe_feature_name}_distribution.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved plot: {plot_filename.name}")
            total_plots += 1
            
        except Exception as e:
            logger.error(f"Error plotting {csv_file.name}: {e}")
            skipped_plots += 1
            continue
    
    logger.info(f"Plotting complete! Created {total_plots} plots, skipped {skipped_plots}")

def create_summary_plot():
    """
    Create a summary plot showing percentage of valid values per feature (aggregated across all bands).
    """
    
    script_dir = Path(__file__).parent
    consolidated_dir = script_dir / 'consolidated_features'
    plots_dir = script_dir / 'feature_plots'
    
    if not consolidated_dir.exists():
        logger.error("consolidated_features directory not found")
        return
    
    csv_files = list(consolidated_dir.glob('*.csv'))
    if not csv_files:
        logger.warning("No CSV files found")
        return
    
    # Collect data for summary
    summary_data = []
    
    for csv_file in csv_files:
        try:
            feature_name = csv_file.stem
            df = pd.read_csv(csv_file)
            
            band_columns = [col for col in df.columns if col.startswith('band_')]
            
            if not band_columns:
                continue
            
            # Calculate completeness across all bands for this feature
            total_possible_values = len(df) * len(band_columns)
            total_valid_values = 0
            
            for band_col in band_columns:
                total_valid_values += df[band_col].count()  # Count non-NaN values
            
            completeness_percentage = (total_valid_values / total_possible_values) * 100 if total_possible_values > 0 else 0
            
            summary_data.append({
                'feature': feature_name,
                'total_records': len(df),
                'total_bands': len(band_columns),
                'valid_values': total_valid_values,
                'possible_values': total_possible_values,
                'completeness_percentage': completeness_percentage
            })
                
        except Exception as e:
            logger.error(f"Error processing {csv_file.name} for summary: {e}")
            continue
    
    if not summary_data:
        logger.warning("No data collected for summary plot")
        return
    
    # Create summary DataFrame and sort by completeness
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('completeness_percentage', ascending=True)
    
    # Create horizontal bar plot
    plt.figure(figsize=(12, max(8, len(summary_df) * 0.4)))
    
    # Color bars based on completeness level
    colors = []
    for pct in summary_df['completeness_percentage']:
        if pct >= 80:
            colors.append('green')
        elif pct >= 50:
            colors.append('orange') 
        else:
            colors.append('red')
    
    bars = plt.barh(summary_df['feature'], summary_df['completeness_percentage'], color=colors, alpha=0.7)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, summary_df['completeness_percentage'])):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.xlabel('Data Completeness (%)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Data Completeness (All Bands Combined)', fontsize=16, fontweight='bold')
    plt.xlim(0, 105)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='≥80% complete'),
        Patch(facecolor='orange', alpha=0.7, label='50-80% complete'),
        Patch(facecolor='red', alpha=0.7, label='<50% complete')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_filename = plots_dir / 'feature_completeness_summary.png'
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved summary plot: {summary_filename.name}")
    
    # Print summary statistics
    print("\n=== FEATURE COMPLETENESS SUMMARY ===")
    print(f"Total features analyzed: {len(summary_df)}")
    high_completeness = (summary_df['completeness_percentage'] >= 80).sum()
    medium_completeness = ((summary_df['completeness_percentage'] >= 50) & (summary_df['completeness_percentage'] < 80)).sum()
    low_completeness = (summary_df['completeness_percentage'] < 50).sum()
    
    print(f"High completeness (≥80%): {high_completeness}")
    print(f"Medium completeness (50-80%): {medium_completeness}")
    print(f"Low completeness (<50%): {low_completeness}")
    print("\nTop 10 most complete features:")
    top_features = summary_df.nlargest(10, 'completeness_percentage')
    for _, row in top_features.iterrows():
        print(f"  {row['feature']:<30}: {row['completeness_percentage']:.1f}%")

if __name__ == "__main__":
    print("=== Feature Distribution Plotting Script ===")
    print("1. Creating individual feature distribution plots...")
    plot_feature_distributions()
    
    print("\n2. Creating data completeness summary...")
    create_summary_plot()
    
    print("\nAll plots saved to feature_plots/ directory!")