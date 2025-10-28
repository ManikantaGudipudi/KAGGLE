"""
Feature Correlation Analysis Script
====================================
This script computes multiple types of correlations between features and target
to understand feature importance and relationships.

Methods:
1. Pearson Correlation (linear relationships)
2. Spearman Correlation (monotonic relationships)
3. Kendall Tau (rank-based)
4. Mutual Information (non-linear relationships)
5. ANOVA F-statistic (categorical features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, f_regression
from scipy.stats import spearmanr, kendalltau, pointbiserialr
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)


def load_and_prepare_data(train_path='train.csv', test_split_size=0.2):
    """
    Load and prepare data for correlation analysis.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_split_size : float
        Proportion of data to use for test split (analysis on train only)
    
    Returns:
    --------
    X_train : DataFrame
        Training features (encoded)
    y_train : Series
        Training target
    X_test : DataFrame
        Test features (for validation)
    y_test : Series
        Test target (for validation)
    feature_info : dict
        Information about feature types
    """
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    # Load data
    df = pd.read_csv(train_path)
    print(f"Data loaded: {df.shape}")
    
    # Split features and target
    y = df['accident_risk']
    X = df.drop(['accident_risk', 'id'], axis=1)
    
    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
    
    # Split into train and test (for analysis)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_size, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Identify feature types BEFORE encoding
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    feature_info = {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'all_features': list(X_train.columns)
    }
    
    print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Store original categorical values for later analysis
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    
    # Encode categorical features
    print("\n" + "-"*80)
    print("ENCODING CATEGORICAL FEATURES")
    print("-"*80)
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
        print(f"{col}: {len(le.classes_)} unique values -> {le.classes_[:5]}...")
    
    feature_info['label_encoders'] = label_encoders
    feature_info['X_train_original'] = X_train_original
    feature_info['X_test_original'] = X_test_original
    
    return X_train, y_train, X_test, y_test, feature_info


def compute_pearson_correlation(X_train, y_train):
    """
    Compute Pearson correlation (linear relationships).
    """
    print("\n" + "="*80)
    print("1. PEARSON CORRELATION (Linear Relationships)")
    print("="*80)
    print("Measures: Linear correlation between -1 and 1")
    print("Best for: Continuous variables, linear relationships")
    
    correlations = []
    for col in X_train.columns:
        corr = X_train[col].corr(y_train)
        correlations.append({
            'feature': col,
            'pearson_corr': corr,
            'abs_pearson_corr': abs(corr)
        })
    
    df_corr = pd.DataFrame(correlations).sort_values('abs_pearson_corr', ascending=False)
    
    print("\nPearson Correlation Results:")
    print("-"*80)
    for idx, row in df_corr.iterrows():
        print(f"{row['feature']:30} : {row['pearson_corr']:8.4f}")
    
    return df_corr


def compute_spearman_correlation(X_train, y_train):
    """
    Compute Spearman correlation (monotonic relationships).
    """
    print("\n" + "="*80)
    print("2. SPEARMAN CORRELATION (Monotonic Relationships)")
    print("="*80)
    print("Measures: Rank-based correlation (captures non-linear monotonic relationships)")
    print("Best for: Ordinal variables, non-linear monotonic relationships")
    
    correlations = []
    for col in X_train.columns:
        corr, pval = spearmanr(X_train[col], y_train)
        correlations.append({
            'feature': col,
            'spearman_corr': corr,
            'abs_spearman_corr': abs(corr),
            'p_value': pval
        })
    
    df_corr = pd.DataFrame(correlations).sort_values('abs_spearman_corr', ascending=False)
    
    print("\nSpearman Correlation Results:")
    print("-"*80)
    for idx, row in df_corr.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature']:30} : {row['spearman_corr']:8.4f} {sig}")
    
    return df_corr


def compute_kendall_tau(X_train, y_train):
    """
    Compute Kendall's Tau (rank correlation).
    """
    print("\n" + "="*80)
    print("3. KENDALL'S TAU (Rank Correlation)")
    print("="*80)
    print("Measures: Concordance-based rank correlation")
    print("Best for: Ordinal data, small sample sizes, robust to outliers")
    
    correlations = []
    for col in X_train.columns:
        corr, pval = kendalltau(X_train[col], y_train)
        correlations.append({
            'feature': col,
            'kendall_tau': corr,
            'abs_kendall_tau': abs(corr),
            'p_value': pval
        })
    
    df_corr = pd.DataFrame(correlations).sort_values('abs_kendall_tau', ascending=False)
    
    print("\nKendall's Tau Results:")
    print("-"*80)
    for idx, row in df_corr.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature']:30} : {row['kendall_tau']:8.4f} {sig}")
    
    return df_corr


def compute_mutual_information(X_train, y_train):
    """
    Compute Mutual Information (captures non-linear relationships).
    """
    print("\n" + "="*80)
    print("4. MUTUAL INFORMATION (Non-linear Relationships)")
    print("="*80)
    print("Measures: Amount of information shared between variables")
    print("Best for: Any type of relationship (linear or non-linear)")
    print("\nComputing... (this may take a moment)")
    
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    
    correlations = []
    for col, mi_score in zip(X_train.columns, mi_scores):
        correlations.append({
            'feature': col,
            'mutual_info': mi_score
        })
    
    df_corr = pd.DataFrame(correlations).sort_values('mutual_info', ascending=False)
    
    print("\nMutual Information Results:")
    print("-"*80)
    for idx, row in df_corr.iterrows():
        print(f"{row['feature']:30} : {row['mutual_info']:8.4f}")
    
    return df_corr


def compute_f_statistic(X_train, y_train):
    """
    Compute ANOVA F-statistic (univariate linear regression).
    """
    print("\n" + "="*80)
    print("5. ANOVA F-STATISTIC (Univariate Linear Regression)")
    print("="*80)
    print("Measures: Linear dependency between each feature and target")
    print("Best for: Feature selection, understanding linear importance")
    
    f_scores, p_values = f_regression(X_train, y_train)
    
    correlations = []
    for col, f_score, p_val in zip(X_train.columns, f_scores, p_values):
        correlations.append({
            'feature': col,
            'f_statistic': f_score,
            'p_value': p_val
        })
    
    df_corr = pd.DataFrame(correlations).sort_values('f_statistic', ascending=False)
    
    print("\nF-Statistic Results:")
    print("-"*80)
    for idx, row in df_corr.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature']:30} : F={row['f_statistic']:10.2f} {sig}")
    
    return df_corr


def create_comprehensive_comparison(pearson_df, spearman_df, kendall_df, mi_df, f_df):
    """
    Create a comprehensive comparison table of all correlation methods.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE CORRELATION COMPARISON")
    print("="*80)
    
    # Merge all dataframes
    comparison = pearson_df[['feature', 'pearson_corr', 'abs_pearson_corr']].copy()
    comparison = comparison.merge(
        spearman_df[['feature', 'spearman_corr', 'abs_spearman_corr']], 
        on='feature'
    )
    comparison = comparison.merge(
        kendall_df[['feature', 'kendall_tau', 'abs_kendall_tau']], 
        on='feature'
    )
    comparison = comparison.merge(
        mi_df[['feature', 'mutual_info']], 
        on='feature'
    )
    comparison = comparison.merge(
        f_df[['feature', 'f_statistic']], 
        on='feature'
    )
    
    # Calculate average rank across all methods
    comparison['avg_abs_rank'] = (
        comparison['abs_pearson_corr'].rank(ascending=False) +
        comparison['abs_spearman_corr'].rank(ascending=False) +
        comparison['abs_kendall_tau'].rank(ascending=False) +
        comparison['mutual_info'].rank(ascending=False) +
        comparison['f_statistic'].rank(ascending=False)
    ) / 5
    
    comparison = comparison.sort_values('avg_abs_rank')
    
    print("\nAll Methods Comparison (sorted by average rank):")
    print("-"*120)
    print(f"{'Feature':<25} {'Pearson':>8} {'Spearman':>8} {'Kendall':>8} {'MI':>8} {'F-Stat':>10} {'Avg Rank':>9}")
    print("-"*120)
    
    for idx, row in comparison.iterrows():
        print(f"{row['feature']:<25} {row['pearson_corr']:8.4f} {row['spearman_corr']:8.4f} "
              f"{row['kendall_tau']:8.4f} {row['mutual_info']:8.4f} {row['f_statistic']:10.2f} "
              f"{row['avg_abs_rank']:9.2f}")
    
    return comparison


def visualize_correlations(comparison, output_file='correlation_comparison.png'):
    """
    Create visualization of different correlation methods.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Correlation Analysis - Multiple Methods', fontsize=16, fontweight='bold')
    
    # 1. Pearson Correlation
    ax = axes[0, 0]
    data = comparison.sort_values('abs_pearson_corr', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in data['pearson_corr']]
    ax.barh(data['feature'], data['pearson_corr'], color=colors, alpha=0.7)
    ax.set_xlabel('Pearson Correlation')
    ax.set_title('1. Pearson (Linear)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Spearman Correlation
    ax = axes[0, 1]
    data = comparison.sort_values('abs_spearman_corr', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in data['spearman_corr']]
    ax.barh(data['feature'], data['spearman_corr'], color=colors, alpha=0.7)
    ax.set_xlabel('Spearman Correlation')
    ax.set_title('2. Spearman (Monotonic)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Kendall's Tau
    ax = axes[0, 2]
    data = comparison.sort_values('abs_kendall_tau', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in data['kendall_tau']]
    ax.barh(data['feature'], data['kendall_tau'], color=colors, alpha=0.7)
    ax.set_xlabel("Kendall's Tau")
    ax.set_title('3. Kendall (Rank)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Mutual Information
    ax = axes[1, 0]
    data = comparison.sort_values('mutual_info', ascending=True)
    ax.barh(data['feature'], data['mutual_info'], color='blue', alpha=0.7)
    ax.set_xlabel('Mutual Information')
    ax.set_title('4. Mutual Information (Non-linear)')
    ax.grid(axis='x', alpha=0.3)
    
    # 5. F-Statistic
    ax = axes[1, 1]
    data = comparison.sort_values('f_statistic', ascending=True)
    ax.barh(data['feature'], data['f_statistic'], color='purple', alpha=0.7)
    ax.set_xlabel('F-Statistic')
    ax.set_title('5. ANOVA F-Statistic')
    ax.grid(axis='x', alpha=0.3)
    
    # 6. Heatmap of normalized correlations
    ax = axes[1, 2]
    heatmap_data = comparison[['feature', 'abs_pearson_corr', 'abs_spearman_corr', 
                                'abs_kendall_tau', 'mutual_info']].set_index('feature')
    # Normalize each column to 0-1 range
    heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    heatmap_data_norm.columns = ['Pearson', 'Spearman', 'Kendall', 'MI']
    
    sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Normalized Score'})
    ax.set_title('6. Normalized Comparison (0-1 scale)')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{output_file}'")
    
    return fig


def save_results(comparison, output_file='correlation_analysis_results.csv'):
    """
    Save comprehensive results to CSV.
    """
    comparison.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")


def feature_selection_recommendation(comparison, threshold_percentile=75):
    """
    Recommend features based on correlation analysis.
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION RECOMMENDATIONS")
    print("="*80)
    
    # Features that rank in top percentile across all methods
    threshold_rank = comparison['avg_abs_rank'].quantile(threshold_percentile / 100)
    
    recommended_features = comparison[comparison['avg_abs_rank'] <= threshold_rank]['feature'].tolist()
    
    print(f"\nTop {threshold_percentile}th percentile features (avg rank <= {threshold_rank:.2f}):")
    print("-"*80)
    for feat in recommended_features:
        row = comparison[comparison['feature'] == feat].iloc[0]
        print(f"✓ {feat:25} (rank: {row['avg_abs_rank']:.2f})")
    
    print(f"\nRecommended to keep: {len(recommended_features)} features")
    print(f"Recommended to remove: {len(comparison) - len(recommended_features)} features")
    
    return recommended_features


if __name__ == "__main__":
    print("="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)
    print("Analyzing feature correlations using multiple methods...")
    print()
    
    # Load and prepare data
    X_train, y_train, X_test, y_test, feature_info = load_and_prepare_data()
    
    # Compute different types of correlations
    pearson_df = compute_pearson_correlation(X_train, y_train)
    spearman_df = compute_spearman_correlation(X_train, y_train)
    kendall_df = compute_kendall_tau(X_train, y_train)
    mi_df = compute_mutual_information(X_train, y_train)
    f_df = compute_f_statistic(X_train, y_train)
    
    # Create comprehensive comparison
    comparison = create_comprehensive_comparison(pearson_df, spearman_df, kendall_df, mi_df, f_df)
    
    # Visualize
    visualize_correlations(comparison)
    
    # Save results
    save_results(comparison)
    
    # Feature selection recommendations
    recommended_features = feature_selection_recommendation(comparison, threshold_percentile=60)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Files created:")
    print("  ✓ correlation_comparison.png - Visualization")
    print("  ✓ correlation_analysis_results.csv - Detailed results")
    print("="*80)

