import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def combine_log_and_lightcurves(log_file, output_file, is_train=True):
    """
    Combine log file with lightcurve data from all splits.
    Creates 6 rows per object (one for each filter) with time series data as lists.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file (train_log.csv or test_log.csv)
    output_file : str
        Path to save the combined output (train.csv or test.csv)
    is_train : bool
        Whether this is training data (includes target column)
    """
    
    print(f"Loading {log_file}...")
    log_df = pd.read_csv(log_file)
    
    print(f"Total objects: {len(log_df)}")
    
    # Prepare the result list
    result_rows = []
    
    # Filters to process
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    
    # Process each object
    for idx, row in tqdm(log_df.iterrows(), total=len(log_df), desc="Processing objects"):
        object_id = row['object_id']
        split_folder = row['split']
        
        # Construct path to the lightcurve file
        base_path = Path(log_file).parent
        lightcurve_file = base_path / split_folder / ('train_full_lightcurves.csv' if is_train else 'test_full_lightcurves.csv')
        
        # Load the lightcurve data for this object
        try:
            # Read the lightcurve file
            lightcurve_df = pd.read_csv(lightcurve_file)
            
            # Filter for this specific object
            object_lc = lightcurve_df[lightcurve_df['object_id'] == object_id]
            
            if len(object_lc) == 0:
                print(f"Warning: No lightcurve data found for {object_id} in {split_folder}")
                continue
            
            # Process each filter
            for filt in filters:
                # Filter data for this specific filter
                filter_data = object_lc[object_lc['Filter'] == filt]
                
                # Remove rows with NaN flux values
                filter_data = filter_data[filter_data['Flux'].notna()]
                
                # Sort by time
                filter_data = filter_data.sort_values('Time (MJD)')
                
                # Create lists of time, flux, and flux_err
                time_list = filter_data['Time (MJD)'].tolist()
                flux_list = filter_data['Flux'].tolist()
                flux_err_list = filter_data['Flux_err'].tolist()
                
                # Create row with log metadata and filter data
                new_row = {
                    'object_id': object_id,
                    'Z': row['Z'],
                    'Z_err': row['Z_err'],
                    'EBV': row['EBV'],
                    'SpecType': row['SpecType'] if 'SpecType' in row else '',
                    'English Translation': row['English Translation'],
                    'split': split_folder,
                    'Filter': filt,
                    'Time': time_list,
                    'Flux': flux_list,
                    'Flux_err': flux_err_list,
                    'num_observations': len(time_list)
                }
                
                # Add target column for training data
                if is_train and 'target' in row:
                    new_row['target'] = row['target']
                
                result_rows.append(new_row)
                
        except FileNotFoundError:
            print(f"Warning: Lightcurve file not found for {object_id} at {lightcurve_file}")
            continue
        except Exception as e:
            print(f"Error processing {object_id}: {str(e)}")
            continue
    
    # Create DataFrame from results
    print(f"\nCreating output DataFrame with {len(result_rows)} rows...")
    result_df = pd.DataFrame(result_rows)
    
    # Reorder columns
    if is_train:
        column_order = ['object_id', 'Z', 'Z_err', 'EBV', 'SpecType', 'English Translation', 
                       'split', 'target', 'Filter', 'Time', 'Flux', 'Flux_err', 'num_observations']
    else:
        column_order = ['object_id', 'Z', 'Z_err', 'EBV', 'SpecType', 'English Translation', 
                       'split', 'Filter', 'Time', 'Flux', 'Flux_err', 'num_observations']
    
    result_df = result_df[column_order]
    
    # Save to CSV
    print(f"Saving to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    print(f"Done! Saved {len(result_df)} rows to {output_file}")
    print(f"Unique objects: {result_df['object_id'].nunique()}")
    print(f"Rows per object: {len(result_df) / result_df['object_id'].nunique():.1f}")
    
    return result_df


if __name__ == "__main__":
    # Set base directory
    base_dir = Path(__file__).parent
    
    print("=" * 80)
    print("Processing TRAINING data...")
    print("=" * 80)
    train_df = combine_log_and_lightcurves(
        log_file=base_dir / 'train_log.csv',
        output_file=base_dir / 'train.csv',
        is_train=True
    )
    
    print("\n" + "=" * 80)
    print("Processing TEST data...")
    print("=" * 80)
    test_df = combine_log_and_lightcurves(
        log_file=base_dir / 'test_log.csv',
        output_file=base_dir / 'test.csv',
        is_train=False
    )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training data: {len(train_df)} rows, {train_df['object_id'].nunique()} unique objects")
    print(f"Test data: {len(test_df)} rows, {test_df['object_id'].nunique()} unique objects")
    print("\nSample of training data:")
    print(train_df.head(10))

