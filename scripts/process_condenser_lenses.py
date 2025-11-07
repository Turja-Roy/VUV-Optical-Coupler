"""
Script to process Edmund Optics condenser lens data from Excel to CSV format.
Extracts relevant optical parameters for raytracing simulations.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def clean_numeric_column(series):
    """Convert string columns with tolerances to numeric values."""
    def extract_numeric(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        # Handle strings like "3.75 ±0.1" or "7.50 +0.0/-0.2"
        if isinstance(val, str):
            # Take the first number before ± or +
            val = val.split('±')[0].split('+')[0].strip()
            try:
                return float(val)
            except ValueError:
                return np.nan
        return np.nan
    
    return series.apply(extract_numeric)


def process_condenser_lenses(input_file, output_file):
    """
    Process Edmund Optics condenser lens Excel file.
    
    Parameters:
    -----------
    input_file : str
        Path to input Excel file
    output_file : str
        Path to output CSV file
    """
    # Read Excel file
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    
    print(f"Found {len(df)} lenses in total")
    
    # Select and rename relevant columns
    processed_df = pd.DataFrame()
    
    # Core identification
    processed_df['Item #'] = df['Stock Number']
    processed_df['Title'] = df['Title']
    
    # Dimensions
    processed_df['Diameter (mm)'] = clean_numeric_column(df['Dia. (mm)'])
    processed_df['Center Thickness (mm)'] = clean_numeric_column(df['CT (mm)'])
    processed_df['Edge Thickness (mm)'] = clean_numeric_column(df['ET (mm)'])
    
    # Optical properties
    processed_df['Focal Length (mm)'] = df['EFL (mm)']
    processed_df['Back Focal Length (mm)'] = df['BFL (mm)']
    processed_df['Numerical Aperture'] = df['NA']
    processed_df['f/#'] = df['f/#']
    
    # Surface properties
    processed_df['Back Surface Shape'] = df['Shape of Back Surface']
    processed_df['Radius R2 (mm)'] = df['Radius R2 (mm)']
    processed_df['Asphere Diameter (mm)'] = df['Diameter of Asphere (mm)']
    
    # Material
    processed_df['Substrate'] = df['Substrate']
    
    # Coating
    processed_df['Coating'] = df['Coating']
    processed_df['Wavelength Range (nm)'] = df['Wavelength Range (nm)']
    
    # Calculate radius of front surface (R1) using lensmaker's equation
    # For thin lens: 1/f = (n-1) * (1/R1 - 1/R2)
    # Assuming typical glass n ~ 1.5 for initial estimate
    n_glass_estimate = 1.5
    
    def calculate_R1(row):
        """Calculate front surface radius using lensmaker's equation."""
        f = row['Focal Length (mm)']
        R2_str = row['Radius R2 (mm)']
        
        if pd.isna(f):
            return np.nan
        
        # Handle R2 values
        if pd.isna(R2_str) or R2_str == 'Plano' or R2_str == '∞':
            # Plano-convex: R2 is infinite
            # 1/f = (n-1)/R1
            R1 = (n_glass_estimate - 1) * f
            return R1
        else:
            try:
                R2 = float(R2_str)
                # General case: 1/f = (n-1) * (1/R1 - 1/R2)
                # R1 = 1 / ((1/f)/(n-1) + 1/R2)
                R1 = 1 / ((1/f) / (n_glass_estimate - 1) + 1/R2)
                return R1
            except (ValueError, ZeroDivisionError):
                return np.nan
    
    processed_df['Radius R1 (mm)'] = processed_df.apply(calculate_R1, axis=1)
    
    # Add lens type classification
    def classify_lens_type(row):
        """Classify lens based on back surface shape."""
        shape = row['Back Surface Shape']
        if pd.isna(shape):
            return 'Unknown'
        elif shape == 'Plano':
            return 'Plano-Convex'
        elif shape == 'Convex':
            return 'Bi-Convex'
        elif shape == 'Aspheric':
            return 'Aspheric'
        else:
            return shape
    
    processed_df['Lens Type'] = processed_df.apply(classify_lens_type, axis=1)
    
    # Sort by lens type and diameter for better organization
    processed_df = processed_df.sort_values(['Lens Type', 'Diameter (mm)', 'Focal Length (mm)'])
    
    # Save to CSV
    print(f"\nSaving processed data to {output_file}...")
    processed_df.to_csv(output_file, index=False, float_format='%.4f')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total lenses processed: {len(processed_df)}")
    print(f"\nLens type breakdown:")
    print(processed_df['Lens Type'].value_counts().to_string())
    print(f"\nDiameter range: {processed_df['Diameter (mm)'].min():.2f} - {processed_df['Diameter (mm)'].max():.2f} mm")
    print(f"Focal length range: {processed_df['Focal Length (mm)'].min():.2f} - {processed_df['Focal Length (mm)'].max():.2f} mm")
    print(f"\nCoating types:")
    print(processed_df['Coating'].value_counts().to_string())
    print("="*60)
    
    return processed_df


def main():
    parser = argparse.ArgumentParser(
        description='Process Edmund Optics condenser lens data from Excel to CSV'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/Edmund-Optics_condenser-lenses-731b9cef.xlsx',
        help='Input Excel file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/Edmund-Optics_condenser-lenses.csv',
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Process the data
    process_condenser_lenses(args.input, args.output)
    print(f"\nDone! Processed data saved to {args.output}")


if __name__ == '__main__':
    main()
