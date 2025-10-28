"""
Feature Correlation Analysis with Feature Engineering
======================================================
This script includes custom feature engineering:
1. Custom ordinal encoding for lighting and weather
2. New composite feature: my_score (multiplication of top features)
3. Recompute all correlations with engineered features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, f_regression
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)


def load_and_prepare_data_with_engineering(train_path='train.csv', test_split_size=0.2):
    """
    Load and prepare data with custom feature engineering.
    """
    print("="*80)
    print("LOADING AND PREPARING DATA WITH FEATURE ENGINEERING")
    print("="*80)
    
    # Load data
    df = pd.read_csv(train_path)
    print(f"Data loaded: {df.shape}")
    
    # Split features and target
    y = df['accident_risk']
    X = df.drop(['accident_risk', 'id'], axis=1)
    
    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")
    
    # Split into train and test (for analysis)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_size, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Identify feature types BEFORE encoding
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    print("\n" + "-"*80)
    print("CUSTOM FEATURE ENGINEERING")
    print("-"*80)
    
    # 1. Custom ordinal encoding for LIGHTING
    print("\n1. Lighting encoding (ordinal based on risk):")
    print("   daylight -> 1 (safest)")
    print("   dim      -> 2 (moderate)")
    print("   night    -> 3 (riskiest)")
    
    lighting_map = {
        'daylight': 1,
        'dim': 2,
        'night': 3
    }
    
    X_train['lighting'] = X_train['lighting'].map(lighting_map)
    X_test['lighting'] = X_test['lighting'].map(lighting_map)
    
    print(f"   Train distribution: {X_train['lighting'].value_counts().sort_index().to_dict()}")
    
    # 2. Custom ordinal encoding for WEATHER
    print("\n2. Weather encoding (ordinal based on risk):")
    print("   clear -> 1 (safest)")
    print("   foggy -> 2 (moderate)")
    print("   rainy -> 3 (riskiest)")
    
    weather_map = {
        'clear': 1,
        'foggy': 2,
        'rainy': 3
    }
    
    X_train['weather'] = X_train['weather'].map(weather_map)
    X_test['weather'] = X_test['weather'].map(weather_map)
    
    print(f"   Train distribution: {X_train['weather'].value_counts().sort_index().to_dict()}")
    
    # 3. Encode remaining categorical features with LabelEncoder
    print("\n3. Encoding remaining categorical features:")
    
    remaining_categorical = [col for col in categorical_cols if col not in ['lighting', 'weather']]
    label_encoders = {}
    
    for col in remaining_categorical:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
        print(f"   {col}: {len(le.classes_)} unique values")
    
    # 4. CREATE NEW COMPOSITE FEATURE: my_score
    print("\n4. Creating composite feature: my_score")
    print("   Formula: curvature × speed_limit × lighting × num_reported_accidents × weather")
    
    X_train['my_score'] = (
        X_train['curvature'] * 
        X_train['speed_limit'] * 
        X_train['lighting'] * 
        X_train['num_reported_accidents'] * 
        X_train['weather']
    )
    
    X_test['my_score'] = (
        X_test['curvature'] * 
        X_test['speed_limit'] * 
        X_test['lighting'] * 
        X_test['num_reported_accidents'] * 
        X_test['weather']
    )
    
    print(f"\n   my_score statistics:")
    print(f"   Min:  {X_train['my_score'].min():.2f}")
    print(f"   Max:  {X_train['my_score'].max():.2f}")
    print(f"   Mean: {X_train['my_score'].mean():.2f}")
    print(f"   Std:  {X_train['my_score'].std():.2f}")
    
    # Summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Total features: {X_train.shape[1]} (including my_score)")
    print(f"Features: {list(X_train.columns)}")
    
    feature_info = {
        'numerical': numerical_cols + ['lighting', 'weather', 'my_score'],
        'categorical': remaining_categorical,
        'all_features': list(X_train.columns),
        'engineered_features': ['lighting', 'weather', 'my_score']
    }
    
    return X_train, y_train, X_test, y_test, feature_info


def compute_all_correlations(X_train, y_train, feature_name_filter=None):
    """
    Compute all correlation methods for given features.
    """
    features_to_analyze = X_train.columns.tolist()
    if feature_name_filter:
        features_to_analyze = [f for f in features_to_analyze if feature_name_filter(f)]
    
    results = []
    
    for col in features_to_analyze:
        # Pearson
        pearson_corr = X_train[col].corr(y_train)
        
        # Spearman
        spearman_corr, spearman_pval = spearmanr(X_train[col], y_train)
        
        # Kendall
        kendall_corr, kendall_pval = kendalltau(X_train[col], y_train)
        
        results.append({
            'feature': col,
            'pearson': pearson_corr,
            'abs_pearson': abs(pearson_corr),
            'spearman': spearman_corr,
            'abs_spearman': abs(spearman_corr),
            'kendall': kendall_corr,
            'abs_kendall': abs(kendall_corr),
        })
    
    # Mutual Information (computed once for all features)
    mi_scores = mutual_info_regression(
        X_train[features_to_analyze], 
        y_train, 
        random_state=42
    )
    
    # F-statistic
    f_scores, f_pvals = f_regression(X_train[features_to_analyze], y_train)
    
    # Add MI and F-stat to results
    for i, row in enumerate(results):
        row['mutual_info'] = mi_scores[i]
        row['f_statistic'] = f_scores[i]
        row['f_pvalue'] = f_pvals[i]
    
    df_results = pd.DataFrame(results)
    
    # Calculate average rank
    df_results['avg_rank'] = (
        df_results['abs_pearson'].rank(ascending=False) +
        df_results['abs_spearman'].rank(ascending=False) +
        df_results['abs_kendall'].rank(ascending=False) +
        df_results['mutual_info'].rank(ascending=False) +
        df_results['f_statistic'].rank(ascending=False)
    ) / 5
    
    return df_results.sort_values('avg_rank')


def print_correlation_results(df_results, title="CORRELATION RESULTS"):
    """
    Print correlation results in a formatted table.
    """
    print("\n" + "="*120)
    print(title)
    print("="*120)
    print(f"{'Feature':<30} {'Pearson':>8} {'Spearman':>8} {'Kendall':>8} {'MI':>8} {'F-Stat':>10} {'Avg Rank':>9}")
    print("-"*120)
    
    for idx, row in df_results.iterrows():
        sig = "***" if row['f_pvalue'] < 0.001 else "**" if row['f_pvalue'] < 0.01 else "*" if row['f_pvalue'] < 0.05 else ""
        print(f"{row['feature']:<30} {row['pearson']:8.4f} {row['spearman']:8.4f} "
              f"{row['kendall']:8.4f} {row['mutual_info']:8.4f} {row['f_statistic']:10.2f} "
              f"{row['avg_rank']:9.2f} {sig}")


def highlight_my_score(df_results):
    """
    Highlight my_score performance compared to individual features.
    """
    print("\n" + "="*120)
    print("SPOTLIGHT: my_score vs Individual Features")
    print("="*120)
    
    my_score_row = df_results[df_results['feature'] == 'my_score']
    
    if len(my_score_row) == 0:
        print("my_score feature not found!")
        return
    
    my_score_row = my_score_row.iloc[0]
    
    print(f"\nmy_score Performance:")
    print(f"  Pearson Correlation: {my_score_row['pearson']:.4f}")
    print(f"  Spearman Correlation: {my_score_row['spearman']:.4f}")
    print(f"  Kendall Tau: {my_score_row['kendall']:.4f}")
    print(f"  Mutual Information: {my_score_row['mutual_info']:.4f}")
    print(f"  F-Statistic: {my_score_row['f_statistic']:.2f}")
    print(f"  Average Rank: {my_score_row['avg_rank']:.2f}")
    
    # Compare with component features
    print("\nComparison with Component Features:")
    print("-"*80)
    
    component_features = ['curvature', 'speed_limit', 'lighting', 'num_reported_accidents', 'weather']
    
    print(f"{'Feature':<30} {'Pearson':>10} {'Rank':>8} {'vs my_score':>15}")
    print("-"*80)
    
    for feat in component_features:
        feat_row = df_results[df_results['feature'] == feat]
        if len(feat_row) > 0:
            feat_row = feat_row.iloc[0]
            comparison = "Better" if feat_row['pearson'] > my_score_row['pearson'] else "Worse"
            diff = feat_row['pearson'] - my_score_row['pearson']
            print(f"{feat:<30} {feat_row['pearson']:10.4f} {feat_row['avg_rank']:8.2f} "
                  f"{comparison:>8} ({diff:+.4f})")
    
    print(f"{'my_score (composite)':<30} {my_score_row['pearson']:10.4f} {my_score_row['avg_rank']:8.2f}")
    
    # Analysis
    print("\n" + "-"*80)
    print("ANALYSIS:")
    
    if my_score_row['avg_rank'] <= 5:
        print(f"✓ my_score ranks #{int(my_score_row['avg_rank'])} overall - EXCELLENT composite feature!")
    elif my_score_row['avg_rank'] <= 10:
        print(f"✓ my_score ranks #{int(my_score_row['avg_rank'])} overall - Good composite feature")
    else:
        print(f"⚠ my_score ranks #{int(my_score_row['avg_rank'])} overall - May not add value")
    
    best_component = df_results[df_results['feature'].isin(component_features)].iloc[0]
    
    if my_score_row['pearson'] > best_component['pearson']:
        print(f"✓ my_score has stronger correlation than best component ({best_component['feature']})")
    else:
        print(f"⚠ Best component ({best_component['feature']}) has stronger correlation than my_score")


def visualize_with_my_score(df_results, output_file='correlation_comparison_engineered.png'):
    """
    Create visualization highlighting my_score.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Correlation Analysis - With Engineered Features', fontsize=16, fontweight='bold')
    
    # 1. Pearson Correlation - Highlight my_score
    ax = axes[0, 0]
    data = df_results.sort_values('abs_pearson', ascending=True)
    colors = ['red' if x == 'my_score' else ('green' if row['pearson'] > 0 else 'blue') 
              for x, row in zip(data['feature'], data.to_dict('records'))]
    ax.barh(data['feature'], data['pearson'], color=colors, alpha=0.7)
    ax.set_xlabel('Pearson Correlation')
    ax.set_title('Pearson Correlation (my_score in RED)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Average Rank - Show my_score position
    ax = axes[0, 1]
    data = df_results.sort_values('avg_rank', ascending=False)
    colors = ['red' if x == 'my_score' else 'steelblue' for x in data['feature']]
    ax.barh(data['feature'], data['avg_rank'], color=colors, alpha=0.7)
    ax.set_xlabel('Average Rank (lower is better)')
    ax.set_title('Overall Ranking (my_score in RED)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_xaxis()  # Lower rank is better
    
    # 3. Top 10 Features by Pearson
    ax = axes[1, 0]
    top10 = df_results.nsmallest(10, 'avg_rank')
    colors = ['red' if x == 'my_score' else 'green' for x in top10['feature']]
    ax.bar(range(len(top10)), top10['pearson'], color=colors, alpha=0.7)
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10['feature'], rotation=45, ha='right')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Top 10 Features by Average Rank')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Comparison: my_score vs components
    ax = axes[1, 1]
    component_features = ['curvature', 'speed_limit', 'lighting', 'num_reported_accidents', 'weather', 'my_score']
    comp_data = df_results[df_results['feature'].isin(component_features)].sort_values('pearson', ascending=True)
    colors = ['red' if x == 'my_score' else 'orange' for x in comp_data['feature']]
    ax.barh(comp_data['feature'], comp_data['pearson'], color=colors, alpha=0.7)
    ax.set_xlabel('Pearson Correlation')
    ax.set_title('my_score vs Component Features')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{output_file}'")
    
    return fig


if __name__ == "__main__":
    print("="*80)
    print("FEATURE CORRELATION ANALYSIS WITH FEATURE ENGINEERING")
    print("="*80)
    print()
    
    # Load and prepare data with custom engineering
    X_train, y_train, X_test, y_test, feature_info = load_and_prepare_data_with_engineering()
    
    # Compute all correlations
    print("\n" + "="*80)
    print("COMPUTING CORRELATIONS FOR ALL FEATURES")
    print("="*80)
    print("(This may take a moment...)")
    
    df_results = compute_all_correlations(X_train, y_train)
    
    # Print results
    print_correlation_results(df_results, "ALL FEATURES - CORRELATION RESULTS")
    
    # Highlight my_score
    highlight_my_score(df_results)
    
    # Show engineered features specifically
    print("\n" + "="*80)
    print("ENGINEERED FEATURES ONLY")
    print("="*80)
    
    engineered = df_results[df_results['feature'].isin(['lighting', 'weather', 'my_score'])]
    print_correlation_results(engineered, "ENGINEERED FEATURES")
    
    # Visualize
    visualize_with_my_score(df_results)
    
    # Save results
    output_file = 'correlation_analysis_engineered_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nTop 10 Features (by average rank):")
    top10 = df_results.nsmallest(10, 'avg_rank')
    for idx, row in top10.iterrows():
        marker = "⭐" if row['feature'] == 'my_score' else "  "
        print(f"{marker} {int(row['avg_rank']):2d}. {row['feature']:<30} (Pearson: {row['pearson']:.4f})")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Files created:")
    print("  ✓ correlation_comparison_engineered.png - Visualization")
    print("  ✓ correlation_analysis_engineered_results.csv - Detailed results")
    print("="*80)

