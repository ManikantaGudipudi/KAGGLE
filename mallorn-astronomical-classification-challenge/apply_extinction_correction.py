import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

# LSST filter coefficients (R_lambda values)
FILTER_COEFFICIENTS = {
    'u': 4.239,
    'g': 3.303,
    'r': 2.285,
    'i': 1.698,
    'z': 1.263,
    'y': 1.088
}

def apply_extinction_correction(flux_list, R_lambda, EBV):
    """
    Apply Milky Way extinction correction to flux values.
    
    Flux_corrected = Flux * 10^(0.4 * R_lambda * EBV)
    
    Parameters:
    -----------
    flux_list : list
        List of flux values
    R_lambda : float
        Filter coefficient (extinction coefficient for the filter)
    EBV : float
        E(B-V) extinction value
    
    Returns:
    --------
    list : Corrected flux values (NaN values are preserved)
    """
    correction_factor = 10 ** (0.4 * R_lambda * EBV)
    corrected_flux = []
    for flux in flux_list:
        # NaN values remain NaN after multiplication
        try:
            if np.isnan(flux):
                corrected_flux.append(np.nan)
            else:
                corrected_flux.append(flux * correction_factor)
        except (TypeError, ValueError):
            # If flux is not a number, append NaN
            corrected_flux.append(np.nan)
    return corrected_flux

def process_file(input_file, output_file, is_train=True):
    """
    Process CSV file to apply extinction correction and clean columns.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to output CSV file
    is_train : bool
        Whether this is training data (has target column)
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Apply extinction correction to flux values
    print("Applying extinction correction to flux values...")
    corrected_flux_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Get the R_lambda coefficient for this filter
        filter_type = row['Filter']
        R_lambda = FILTER_COEFFICIENTS[filter_type]
        EBV = row['EBV']
        
        # Parse the flux list (stored as string)
        flux_str = row['Flux']
        if pd.isna(flux_str) or flux_str == '' or flux_str == '[]':
            # Handle empty or NaN flux values
            flux_corrected = []
        else:
            # Replace 'nan' with 'None' for ast.literal_eval, then convert back to np.nan
            flux_str_clean = flux_str.replace('nan', 'None')
            flux_list_temp = ast.literal_eval(flux_str_clean)
            # Convert None back to np.nan
            flux_list = [np.nan if x is None else x for x in flux_list_temp]
            # Apply extinction correction (preserves NaN values)
            flux_corrected = apply_extinction_correction(flux_list, R_lambda, EBV)
        
        corrected_flux_list.append(flux_corrected)
    
    # Add corrected flux column
    df['Flux_corrected'] = corrected_flux_list
    
    # Remove specified columns
    columns_to_remove = ['Z_err', 'split', 'SpecType', 'English Translation']
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    print(f"Removing columns: {existing_columns_to_remove}")
    df = df.drop(columns=existing_columns_to_remove)
    
    # Reorder columns
    if is_train:
        column_order = ['object_id', 'Z', 'EBV', 'target', 'Filter', 
                       'Time', 'Flux', 'Flux_corrected', 'Flux_err', 'num_observations']
    else:
        column_order = ['object_id', 'Z', 'EBV', 'Filter', 
                       'Time', 'Flux', 'Flux_corrected', 'Flux_err', 'num_observations']
    
    df = df[column_order]
    
    # Save to output file
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"Done! Final shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    
    # Show sample
    print("\nSample of corrected data:")
    sample_row = df.iloc[0]
    print(f"Object ID: {sample_row['object_id']}")
    print(f"Filter: {sample_row['Filter']}")
    print(f"EBV: {sample_row['EBV']}")
    print(f"R_lambda: {FILTER_COEFFICIENTS[sample_row['Filter']]}")
    original_flux = ast.literal_eval(sample_row['Flux'])
    print(f"Original Flux (first 3): {original_flux[:3]}")
    print(f"Corrected Flux (first 3): {sample_row['Flux_corrected'][:3]}")
    print(f"Correction factor: {10 ** (0.4 * FILTER_COEFFICIENTS[sample_row['Filter']] * sample_row['EBV']):.6f}")
    
    return df

if __name__ == "__main__":
    print("=" * 80)
    print("Processing TRAINING data...")
    print("=" * 80)
    train_df = process_file(
        input_file='train.csv',
        output_file='train_lightcurves_corrected.csv',
        is_train=True
    )
    
    print("\n" + "=" * 80)
    print("Processing TEST data...")
    print("=" * 80)
    test_df = process_file(
        input_file='test.csv',
        output_file='test_lightcurves_corrected.csv',
        is_train=False
    )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training data saved: train_lightcurves_corrected.csv ({len(train_df)} rows)")
    print(f"Test data saved: test_lightcurves_corrected.csv ({len(test_df)} rows)")
    print("\nExtinction correction applied using:")
    print("  Formula: Flux_corrected = Flux * 10^(0.4 * R_lambda * EBV)")
    print("  Filter coefficients (R_lambda):")
    for filter_name, coeff in sorted(FILTER_COEFFICIENTS.items()):
        print(f"    {filter_name}: {coeff}")

