import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FEATURE-TARGET CORRELATION ANALYSIS")
print("="*80)

# Load training data
print("\nLoading training data...")
df = pd.read_csv('train_decisionTree_features.csv')
print(f"Loaded {len(df)} objects with {len(df.columns)} columns")

# Separate features and target
target_col = 'target'
exclude_cols = ['object_id', 'target']

# Only select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"\nTotal columns: {len(df.columns)}")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"Analyzing {len(feature_cols)} numeric features")
print(f"Target distribution: {dict(df[target_col].value_counts().sort_index())}")

# Calculate Pearson correlation
print("\nCalculating Pearson correlations...")
correlations = {}
for col in feature_cols:
    # Skip if all values are NaN or constant
    if df[col].notna().sum() < 2 or df[col].std() == 0:
        correlations[col] = np.nan
        continue
    
    # Calculate correlation, handling NaN values
    valid_mask = df[col].notna() & df[target_col].notna()
    if valid_mask.sum() < 2:
        correlations[col] = np.nan
        continue
    
    corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, target_col])
    correlations[col] = corr

# Create correlation dataframe
corr_df = pd.DataFrame({
    'feature': list(correlations.keys()),
    'correlation': list(correlations.values())
})

# Add absolute correlation for sorting
corr_df['abs_correlation'] = corr_df['correlation'].abs()

# Sort by absolute correlation (descending)
corr_df = corr_df.sort_values('abs_correlation', ascending=False)

# Remove NaN correlations
corr_df_clean = corr_df[corr_df['correlation'].notna()].copy()

print(f"\nSuccessfully calculated correlations for {len(corr_df_clean)} features")
print(f"Features with NaN correlation: {len(corr_df) - len(corr_df_clean)}")

# Display top correlations
print("\n" + "="*80)
print("TOP 30 FEATURES BY ABSOLUTE CORRELATION WITH TARGET")
print("="*80)
print("\nFeature                              Correlation    Abs Corr")
print("-" * 80)
for idx, row in corr_df_clean.head(30).iterrows():
    feat = row['feature']
    corr = row['correlation']
    abs_corr = row['abs_correlation']
    direction = "↑" if corr > 0 else "↓"
    print(f"{feat:40s} {direction} {corr:9.6f}    {abs_corr:.6f}")

# Statistics
print("\n" + "="*80)
print("CORRELATION STATISTICS")
print("="*80)
print(f"Mean absolute correlation: {corr_df_clean['abs_correlation'].mean():.6f}")
print(f"Median absolute correlation: {corr_df_clean['abs_correlation'].median():.6f}")
print(f"Max absolute correlation: {corr_df_clean['abs_correlation'].max():.6f}")
print(f"Min absolute correlation: {corr_df_clean['abs_correlation'].min():.6f}")

# Count features by correlation strength
strong = (corr_df_clean['abs_correlation'] >= 0.1).sum()
moderate = ((corr_df_clean['abs_correlation'] >= 0.05) & (corr_df_clean['abs_correlation'] < 0.1)).sum()
weak = (corr_df_clean['abs_correlation'] < 0.05).sum()

print(f"\nFeatures by correlation strength:")
print(f"  Strong (|r| >= 0.10): {strong}")
print(f"  Moderate (0.05 <= |r| < 0.10): {moderate}")
print(f"  Weak (|r| < 0.05): {weak}")

# Positive vs negative correlations
positive = (corr_df_clean['correlation'] > 0).sum()
negative = (corr_df_clean['correlation'] < 0).sum()
print(f"\nCorrelation direction:")
print(f"  Positive correlations: {positive}")
print(f"  Negative correlations: {negative}")

# Save correlation results
output_file = 'feature_target_correlations.csv'
corr_df_clean.to_csv(output_file, index=False)
print(f"\n✓ Saved correlations to: {output_file}")

# Feature type analysis
print("\n" + "="*80)
print("CORRELATION BY FEATURE TYPE")
print("="*80)

feature_types = {
    'metadata': ['Z', 'log1pZ', 'EBV'],
    'presence': [col for col in corr_df_clean['feature'] if '_present' in col],
    'cadence': [col for col in corr_df_clean['feature'] if any(x in col for x in ['_n', '_span', '_dt_med', '_short'])],
    'flux_stats': [col for col in corr_df_clean['feature'] if any(x in col for x in ['_flux_mean', '_flux_std', '_flux_mad', '_flux_iqr', '_flux_min', '_flux_max', '_flux_amp'])],
    'quality': [col for col in corr_df_clean['feature'] if any(x in col for x in ['_neg_frac', '_snr'])],
    'shape': [col for col in corr_df_clean['feature'] if any(x in col for x in ['_t_peak', '_rise_rate'])],
    'global': [col for col in corr_df_clean['feature'] if col.startswith('global_') or col.startswith('n_filters') or col.startswith('total_')]
}

for ftype, feats in feature_types.items():
    feats_in_data = [f for f in feats if f in corr_df_clean['feature'].values]
    if len(feats_in_data) > 0:
        mean_abs_corr = corr_df_clean[corr_df_clean['feature'].isin(feats_in_data)]['abs_correlation'].mean()
        max_abs_corr = corr_df_clean[corr_df_clean['feature'].isin(feats_in_data)]['abs_correlation'].max()
        top_feat = corr_df_clean[corr_df_clean['feature'].isin(feats_in_data)].iloc[0]
        print(f"\n{ftype.upper()}:")
        print(f"  Features: {len(feats_in_data)}")
        print(f"  Mean |correlation|: {mean_abs_corr:.6f}")
        print(f"  Max |correlation|: {max_abs_corr:.6f}")
        print(f"  Top feature: {top_feat['feature']} (r={top_feat['correlation']:.6f})")

# Create visualization
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Top 20 features bar plot
top_20 = corr_df_clean.head(20).copy()
colors = ['red' if x < 0 else 'green' for x in top_20['correlation']]
ax1 = axes[0, 0]
ax1.barh(range(len(top_20)), top_20['correlation'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['feature'], fontsize=8)
ax1.set_xlabel('Correlation with Target', fontsize=10)
ax1.set_title('Top 20 Features by Absolute Correlation', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# 2. Correlation distribution histogram
ax2 = axes[0, 1]
ax2.hist(corr_df_clean['correlation'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Correlation with Target', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Distribution of Feature Correlations', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Absolute correlation distribution
ax3 = axes[1, 0]
ax3.hist(corr_df_clean['abs_correlation'], bins=50, color='orange', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Absolute Correlation', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Distribution of Absolute Correlations', fontsize=12, fontweight='bold')
ax3.axvline(x=corr_df_clean['abs_correlation'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {corr_df_clean["abs_correlation"].mean():.4f}')
ax3.axvline(x=corr_df_clean['abs_correlation'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Median: {corr_df_clean["abs_correlation"].median():.4f}')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Feature type comparison
ax4 = axes[1, 1]
type_means = []
type_names = []
for ftype, feats in feature_types.items():
    feats_in_data = [f for f in feats if f in corr_df_clean['feature'].values]
    if len(feats_in_data) > 0:
        mean_corr = corr_df_clean[corr_df_clean['feature'].isin(feats_in_data)]['abs_correlation'].mean()
        type_means.append(mean_corr)
        type_names.append(ftype)

type_df = pd.DataFrame({'type': type_names, 'mean_abs_corr': type_means}).sort_values('mean_abs_corr', ascending=True)
ax4.barh(range(len(type_df)), type_df['mean_abs_corr'], color='purple', alpha=0.7)
ax4.set_yticks(range(len(type_df)))
ax4.set_yticklabels(type_df['type'], fontsize=10)
ax4.set_xlabel('Mean Absolute Correlation', fontsize=10)
ax4.set_title('Feature Type Performance', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plot_file = 'feature_correlations_analysis.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print(f"  1. {output_file} - Detailed correlation table")
print(f"  2. {plot_file} - Visualization of correlations")

