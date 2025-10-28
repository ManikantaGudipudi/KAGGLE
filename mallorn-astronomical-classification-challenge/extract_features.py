import pandas as pd
import numpy as np
import ast
from scipy.stats import theilslopes, iqr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def parse_list_column(val):
    """Parse stringified list to numpy array."""
    if pd.isna(val) or val == '' or val == '[]':
        return np.array([])
    try:
        return np.array(ast.literal_eval(val))
    except:
        return np.array([])

def median_absolute_deviation(x):
    """Compute MAD."""
    if len(x) == 0:
        return np.nan
    return np.median(np.abs(x - np.median(x)))

def compute_per_filter_features(time, flux_corr, flux_raw, flux_err, t0, z, filter_name):
    """
    Compute per-filter features.
    
    Returns dict with keys prefixed by filter_name (e.g., 'r_flux_mean').
    """
    prefix = f"{filter_name}_"
    features = {}
    
    # Check if filter has data
    n_obs = len(time)
    features[f'{prefix}present'] = 1 if n_obs > 0 else 0
    features[f'{prefix}n'] = n_obs
    features[f'{prefix}short'] = 1 if n_obs < 3 else 0
    
    if n_obs == 0:
        # Fill with NaN if no data
        for feat in ['span_rest', 'dt_med', 'flux_mean', 'flux_std', 'flux_mad', 
                     'flux_iqr', 'flux_min', 'flux_max', 'flux_amp', 'neg_frac',
                     'snr_mean', 'snr_p90', 't_peak_rest', 'rise_rate']:
            features[f'{prefix}{feat}'] = np.nan
        return features
    
    # Compute rest-frame time
    t_rest = (time - t0) / (1 + z)
    
    # Cadence features
    if n_obs > 1:
        features[f'{prefix}span_rest'] = t_rest.max() - t_rest.min()
        dt_consecutive = np.diff(t_rest)
        features[f'{prefix}dt_med'] = np.median(dt_consecutive) if len(dt_consecutive) > 0 else np.nan
    else:
        features[f'{prefix}span_rest'] = 0
        features[f'{prefix}dt_med'] = np.nan
    
    # Flux statistics (on Flux_corrected)
    features[f'{prefix}flux_mean'] = np.mean(flux_corr)
    features[f'{prefix}flux_std'] = np.std(flux_corr) if n_obs > 1 else 0
    features[f'{prefix}flux_mad'] = median_absolute_deviation(flux_corr)
    features[f'{prefix}flux_iqr'] = iqr(flux_corr) if n_obs > 1 else 0
    features[f'{prefix}flux_min'] = np.min(flux_corr)
    features[f'{prefix}flux_max'] = np.max(flux_corr)
    features[f'{prefix}flux_amp'] = np.max(flux_corr) - np.min(flux_corr)
    
    # Quality metrics (on raw Flux)
    features[f'{prefix}neg_frac'] = np.sum(flux_raw < 0) / n_obs
    
    # SNR features
    valid_snr = (flux_err > 0) & np.isfinite(flux_err)
    if np.any(valid_snr):
        snr = np.abs(flux_raw[valid_snr]) / flux_err[valid_snr]
        features[f'{prefix}snr_mean'] = np.mean(snr)
        features[f'{prefix}snr_p90'] = np.percentile(snr, 90)
    else:
        features[f'{prefix}snr_mean'] = np.nan
        features[f'{prefix}snr_p90'] = np.nan
    
    # Peak features
    peak_idx = np.argmax(flux_corr)
    features[f'{prefix}t_peak_rest'] = t_rest[peak_idx]
    
    # Rise rate (Theil-Sen slope from first quartile to peak)
    if n_obs >= 4:
        q1_idx = int(n_obs * 0.25)
        peak_idx = np.argmax(flux_corr)
        
        if peak_idx > q1_idx:
            t_rise = t_rest[q1_idx:peak_idx+1]
            f_rise = flux_corr[q1_idx:peak_idx+1]
            
            if len(t_rise) >= 4:
                try:
                    slope, _, _, _ = theilslopes(f_rise, t_rise)
                    features[f'{prefix}rise_rate'] = slope
                except:
                    features[f'{prefix}rise_rate'] = np.nan
            else:
                features[f'{prefix}rise_rate'] = np.nan
        else:
            features[f'{prefix}rise_rate'] = np.nan
    else:
        features[f'{prefix}rise_rate'] = np.nan
    
    return features

def compute_global_features(filter_data, t0, z):
    """
    Compute global features across all filters.
    
    filter_data: dict mapping filter name to dict with keys 'time', 'flux_corr', 'flux_raw', 'flux_err'
    """
    features = {}
    
    # Combine all data
    all_time = []
    all_flux_corr = []
    all_flux_raw = []
    all_flux_err = []
    filters_present = []
    
    for filt, data in filter_data.items():
        if len(data['time']) > 0:
            filters_present.append(filt)
            all_time.append(data['time'])
            all_flux_corr.append(data['flux_corr'])
            all_flux_raw.append(data['flux_raw'])
            all_flux_err.append(data['flux_err'])
    
    features['n_filters_present'] = len(filters_present)
    
    if len(all_time) == 0:
        # No data at all
        features['total_n_obs'] = 0
        features['total_span_rest'] = np.nan
        for feat in ['global_flux_mean', 'global_flux_std', 'global_flux_mad', 
                     'global_flux_iqr', 'global_min_flux', 'global_max_flux', 
                     'global_amp', 'global_neg_frac', 'global_snr_p90',
                     'global_t_peak_rest']:
            features[feat] = np.nan
        features['global_peak_filter'] = ''
        return features
    
    # Concatenate all arrays
    all_time = np.concatenate(all_time)
    all_flux_corr = np.concatenate(all_flux_corr)
    all_flux_raw = np.concatenate(all_flux_raw)
    all_flux_err = np.concatenate(all_flux_err)
    
    # Compute rest-frame time
    t_rest = (all_time - t0) / (1 + z)
    
    # Basic stats
    features['total_n_obs'] = len(all_time)
    features['total_span_rest'] = t_rest.max() - t_rest.min() if len(t_rest) > 1 else 0
    
    # Flux statistics (on Flux_corrected)
    features['global_flux_mean'] = np.mean(all_flux_corr)
    features['global_flux_std'] = np.std(all_flux_corr)
    features['global_flux_mad'] = median_absolute_deviation(all_flux_corr)
    features['global_flux_iqr'] = iqr(all_flux_corr)
    features['global_min_flux'] = np.min(all_flux_corr)
    features['global_max_flux'] = np.max(all_flux_corr)
    features['global_amp'] = np.max(all_flux_corr) - np.min(all_flux_corr)
    
    # Quality metrics
    features['global_neg_frac'] = np.sum(all_flux_raw < 0) / len(all_flux_raw)
    
    # SNR
    valid_snr = (all_flux_err > 0) & np.isfinite(all_flux_err)
    if np.any(valid_snr):
        snr = np.abs(all_flux_raw[valid_snr]) / all_flux_err[valid_snr]
        features['global_snr_p90'] = np.percentile(snr, 90)
    else:
        features['global_snr_p90'] = np.nan
    
    # Global peak
    global_peak_idx = np.argmax(all_flux_corr)
    features['global_t_peak_rest'] = t_rest[global_peak_idx]
    
    # Find which filter has the peak
    cumulative_lens = np.cumsum([len(filter_data[f]['flux_corr']) for f in filters_present])
    filter_idx = np.searchsorted(cumulative_lens, global_peak_idx, side='right')
    features['global_peak_filter'] = filters_present[filter_idx] if filter_idx < len(filters_present) else filters_present[-1]
    
    return features

def extract_features_from_dataset(df, dataset_name, has_target=True):
    """
    Extract features from dataset.
    
    Args:
        df: DataFrame in long format (one row per object_id × Filter)
        dataset_name: string name for reporting
        has_target: whether target column exists
    
    Returns:
        DataFrame with one row per object_id containing all features
    """
    print(f"\n{'='*80}")
    print(f"Extracting features from {dataset_name}")
    print(f"{'='*80}")
    
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    
    # Group by object_id
    grouped = df.groupby('object_id')
    
    results = []
    
    for object_id, group in tqdm(grouped, desc=f"Processing {dataset_name}"):
        obj_features = {'object_id': object_id}
        
        # Get metadata (same for all rows of this object)
        first_row = group.iloc[0]
        Z = first_row['Z']
        EBV = first_row['EBV']
        
        obj_features['Z'] = Z
        obj_features['log1pZ'] = np.log1p(Z)
        obj_features['EBV'] = EBV
        
        if has_target:
            obj_features['target'] = first_row['target']
        
        # Collect data for each filter
        filter_data = {}
        all_times = []
        
        for _, row in group.iterrows():
            filt = row['Filter']
            
            time = parse_list_column(row['Time'])
            flux_corr = parse_list_column(row['Flux_corrected'])
            flux_raw = parse_list_column(row['Flux'])
            flux_err = parse_list_column(row['Flux_err'])
            
            filter_data[filt] = {
                'time': time,
                'flux_corr': flux_corr,
                'flux_raw': flux_raw,
                'flux_err': flux_err
            }
            
            if len(time) > 0:
                all_times.append(time)
        
        # Compute t0 (global minimum time across all filters)
        if len(all_times) > 0:
            t0 = np.min(np.concatenate(all_times))
        else:
            t0 = 0
        
        # Compute per-filter features
        for filt in filters:
            if filt in filter_data:
                data = filter_data[filt]
                per_filter_feats = compute_per_filter_features(
                    data['time'], data['flux_corr'], data['flux_raw'], 
                    data['flux_err'], t0, Z, filt
                )
                obj_features.update(per_filter_feats)
            else:
                # Filter not present - fill with defaults
                prefix = f"{filt}_"
                obj_features[f'{prefix}present'] = 0
                obj_features[f'{prefix}n'] = 0
                obj_features[f'{prefix}short'] = 1
                for feat in ['span_rest', 'dt_med', 'flux_mean', 'flux_std', 'flux_mad', 
                             'flux_iqr', 'flux_min', 'flux_max', 'flux_amp', 'neg_frac',
                             'snr_mean', 'snr_p90', 't_peak_rest', 'rise_rate']:
                    obj_features[f'{prefix}{feat}'] = np.nan
        
        # Compute global features
        global_feats = compute_global_features(filter_data, t0, Z)
        obj_features.update(global_feats)
        
        results.append(obj_features)
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Print report
    print(f"\n{'='*80}")
    print(f"Feature Extraction Report - {dataset_name}")
    print(f"{'='*80}")
    print(f"Number of objects processed: {len(result_df)}")
    print(f"Average filters present per object: {result_df['n_filters_present'].mean():.2f}")
    print(f"\nFilter availability:")
    for filt in filters:
        availability = result_df[f'{filt}_present'].sum() / len(result_df) * 100
        print(f"  {filt}: {availability:.2f}%")
    
    if has_target:
        print(f"\nTarget distribution:")
        print(result_df['target'].value_counts().sort_index())
    
    print(f"\nTotal features extracted: {len(result_df.columns) - 1}")  # -1 for object_id
    
    return result_df

def main():
    print("="*80)
    print("FEATURE EXTRACTION FOR DECISION TREE / ML MODELS")
    print("="*80)
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv('train_lightcurves_corrected.csv')
    print(f"Loaded {len(train_df)} rows")
    
    # Extract training features
    train_features = extract_features_from_dataset(train_df, "Training Set", has_target=True)
    
    # Save training features
    output_file = 'train_decisionTree_features.csv'
    train_features.to_csv(output_file, index=False)
    print(f"\n✓ Saved training features to: {output_file}")
    
    # Load test data
    print("\n" + "="*80)
    print("\nLoading test data...")
    test_df = pd.read_csv('test_lightcurves_corrected.csv')
    print(f"Loaded {len(test_df)} rows")
    
    # Extract test features
    test_features = extract_features_from_dataset(test_df, "Test Set", has_target=False)
    
    # Save test features
    output_file = 'test_decisionTree_features.csv'
    test_features.to_csv(output_file, index=False)
    print(f"\n✓ Saved test features to: {output_file}")
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - train_decisionTree_features.csv ({len(train_features)} objects)")
    print(f"  - test_decisionTree_features.csv ({len(test_features)} objects)")
    
    # Show sample features
    print(f"\nSample features (first 5 columns):")
    print(train_features.iloc[:3, :5])

if __name__ == "__main__":
    main()

