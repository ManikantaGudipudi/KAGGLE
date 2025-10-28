"""
XGBoost Binary Classifier for TDE Detection
Trains on train_decisionTree_features.csv with hyperparameter tuning
Predicts on test_decisionTree_features.csv and generates submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    fbeta_score, 
    confusion_matrix,
    precision_recall_curve,
    classification_report
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("XGBoost TDE CLASSIFICATION - TRAINING & PREDICTION")
print("="*80)

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("\n[1] Loading training data...")
train_df = pd.read_csv('train_decisionTree_features.csv')
print(f"    Loaded {len(train_df)} objects with {len(train_df.columns)} columns")

# Separate features and target
target_col = 'target'
drop_cols = ['object_id', 'target']

# Get numeric features only
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in drop_cols]

X = train_df[feature_cols].values
y = train_df[target_col].values

print(f"    Features: {len(feature_cols)}")
print(f"    Target distribution: {np.bincount(y)} (0=Non-TDE, 1=TDE)")
print(f"    Class imbalance ratio: {np.sum(y==0)/np.sum(y==1):.2f}:1")

# ============================================================================
# STEP 2: Train-Validation Split (Stratified)
# ============================================================================
print("\n[2] Creating stratified train-validation split (80-20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"    Training set: {len(X_train)} samples")
print(f"    Validation set: {len(X_val)} samples")
print(f"    Train target distribution: {np.bincount(y_train)}")
print(f"    Val target distribution: {np.bincount(y_val)}")

# ============================================================================
# STEP 3: Compute scale_pos_weight for Class Imbalance
# ============================================================================
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_pos_weight = num_neg / num_pos

print(f"\n[3] Class balance parameter:")
print(f"    scale_pos_weight = {num_neg}/{num_pos} = {scale_pos_weight:.2f}")

# ============================================================================
# STEP 4: Hyperparameter Grid Search with Cross-Validation
# ============================================================================
print("\n[4] Starting hyperparameter grid search...")
print("    This may take a while...")

# Define parameter grid
param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [200, 400, 800],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.5, 1]
}

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"    Total parameter combinations: {total_combinations}")
print(f"    With 5-fold CV: {total_combinations * 5} model fits")

# Initialize base XGBoost classifier
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',  # Fast histogram-based algorithm
    random_state=42,
    n_jobs=-1
)

# Setup stratified k-fold cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='average_precision',  # PR-AUC
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

# Fit grid search
print("\n    Starting grid search (this will take time)...")
grid_search.fit(X_train, y_train)

# Extract best parameters and score
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print("\n" + "="*80)
print("BEST HYPERPARAMETERS FROM GRID SEARCH")
print("="*80)
for param, value in best_params.items():
    print(f"    {param}: {value}")
print(f"\n    Best CV PR-AUC: {best_cv_score:.6f}")

# ============================================================================
# STEP 5: Train Final Model with Best Parameters
# ============================================================================
print("\n[5] Training final model with best parameters...")

# Create final model with best parameters
final_model = xgb.XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='aucpr',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

# Train on full training set with early stopping on validation set
eval_set = [(X_train, y_train), (X_val, y_val)]
final_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

print("    ✓ Model trained successfully")

# ============================================================================
# STEP 6: Validation Set Evaluation
# ============================================================================
print("\n[6] Evaluating on validation set...")

# Get probability predictions
y_val_proba = final_model.predict_proba(X_val)[:, 1]

# Calculate metrics
val_pr_auc = average_precision_score(y_val, y_val_proba)
val_roc_auc = roc_auc_score(y_val, y_val_proba)

print(f"    Validation PR-AUC: {val_pr_auc:.6f}")
print(f"    Validation ROC-AUC: {val_roc_auc:.6f}")

# ============================================================================
# STEP 7: Find Optimal F2 Threshold
# ============================================================================
print("\n[7] Finding optimal F2 threshold...")

# Scan thresholds
thresholds = np.arange(0.01, 1.0, 0.01)
f2_scores = []

for threshold in thresholds:
    y_pred = (y_val_proba >= threshold).astype(int)
    f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
    f2_scores.append(f2)

# Find best threshold
best_f2_idx = np.argmax(f2_scores)
best_threshold = thresholds[best_f2_idx]
best_f2_score = f2_scores[best_f2_idx]

print(f"    Best F2 threshold: {best_threshold:.3f}")
print(f"    Best F2 score: {best_f2_score:.6f}")

# Make predictions with best threshold
y_val_pred = (y_val_proba >= best_threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
print("\n    Confusion Matrix:")
print(f"    {cm}")
print(f"\n    True Negatives:  {cm[0,0]}")
print(f"    False Positives: {cm[0,1]}")
print(f"    False Negatives: {cm[1,0]}")
print(f"    True Positives:  {cm[1,1]}")

# Classification report
print("\n    Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Non-TDE', 'TDE']))

# ============================================================================
# STEP 8: Save Model
# ============================================================================
print("\n[8] Saving model...")
model_file = 'xgb_mallorn_best.json'
final_model.save_model(model_file)
print(f"    ✓ Model saved to: {model_file}")

# ============================================================================
# STEP 9: Create Visualizations
# ============================================================================
print("\n[9] Creating visualizations...")

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_val, y_val_proba)

plt.figure(figsize=(12, 10))

# Plot 1: Precision-Recall Curve
plt.subplot(2, 2, 1)
plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC={val_pr_auc:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=10)

# Plot 2: F2 Score vs Threshold
plt.subplot(2, 2, 2)
plt.plot(thresholds, f2_scores, linewidth=2, color='green')
plt.axvline(best_threshold, color='red', linestyle='--', 
            label=f'Best threshold={best_threshold:.3f}')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('F2 Score', fontsize=12)
plt.title('F2 Score vs Threshold', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=10)

# Plot 3: Confusion Matrix Heatmap
plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-TDE', 'TDE'],
            yticklabels=['Non-TDE', 'TDE'])
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 4: Feature Importance (Top 20)
plt.subplot(2, 2, 4)
feature_importance = final_model.feature_importances_
feature_names = [feature_cols[i] for i in range(len(feature_cols))]
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(20)

plt.barh(range(len(feat_imp_df)), feat_imp_df['importance'], color='purple', alpha=0.7)
plt.yticks(range(len(feat_imp_df)), feat_imp_df['feature'], fontsize=8)
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plot_file = 'xgboost_evaluation.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"    ✓ Evaluation plots saved to: {plot_file}")

# ============================================================================
# STEP 10: Predict on Test Set
# ============================================================================
print("\n[10] Loading test data and making predictions...")
test_df = pd.read_csv('test_decisionTree_features.csv')
print(f"     Loaded {len(test_df)} test objects")

# Extract features (same columns as training)
X_test = test_df[feature_cols].values

# Make predictions
test_proba = final_model.predict_proba(X_test)[:, 1]
print(f"     Generated predictions for {len(test_proba)} objects")

# ============================================================================
# STEP 11: Create Submission File
# ============================================================================
print("\n[11] Creating submission file...")

# Load sample submission to get the correct format
sample_sub = pd.read_csv('sample_submission.csv')
print(f"     Sample submission format: {list(sample_sub.columns)}")

# Create submission dataframe
submission = pd.DataFrame({
    'object_id': test_df['object_id'],
    'target': test_proba
})

# Save submission
submission_file = 'submission_xgboost.csv'
submission.to_csv(submission_file, index=False)
print(f"     ✓ Submission saved to: {submission_file}")
print(f"     Submission shape: {submission.shape}")
print(f"\n     Prediction statistics:")
print(f"       Mean probability: {test_proba.mean():.4f}")
print(f"       Std probability: {test_proba.std():.4f}")
print(f"       Min probability: {test_proba.min():.4f}")
print(f"       Max probability: {test_proba.max():.4f}")
print(f"       Predictions > 0.5: {(test_proba > 0.5).sum()}")
print(f"       Predictions > {best_threshold:.3f} (best threshold): {(test_proba > best_threshold).sum()}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"\n✓ Best PR-AUC (5-Fold CV):        {best_cv_score:.6f}")
print(f"✓ Validation PR-AUC:              {val_pr_auc:.6f}")
print(f"✓ Validation ROC-AUC:             {val_roc_auc:.6f}")
print(f"✓ Best F2 threshold:              {best_threshold:.3f}")
print(f"✓ F2 score at best threshold:    {best_f2_score:.6f}")
print(f"\n✓ Model saved:                    {model_file}")
print(f"✓ Plots saved:                    {plot_file}")
print(f"✓ Submission file:                {submission_file}")

print("\n" + "="*80)
print("ALL DONE! 🚀")
print("="*80)

