import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A class to handle data preprocessing for train and test datasets.
    Handles categorical encoding, numerical scaling, and feature selection.
    """
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.selected_features = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.feature_correlations = None
        
    def fit_transform(self, train_df, target_col='accident_risk', correlation_threshold=0.1):
        """
        Fit the preprocessor on training data and transform it.
        
        Parameters:
        -----------
        train_df : DataFrame
            Training data with target column
        target_col : str
            Name of the target column
        correlation_threshold : float
            Minimum absolute correlation to keep a feature
            
        Returns:
        --------
        X_train : DataFrame
            Preprocessed features
        y_train : Series
            Target variable
        """
        print("="*80)
        print("FITTING AND TRANSFORMING TRAINING DATA")
        print("="*80)
        
        # Create a copy to avoid modifying original
        df = train_df.copy()
        
        # Split features and target
        y_train = df[target_col]
        X = df.drop([target_col, 'id'], axis=1)
        
        print(f"\nOriginal data shape: {X.shape}")
        print(f"Target variable: {target_col}")
        print(f"Target statistics: min={y_train.min():.3f}, max={y_train.max():.3f}, mean={y_train.mean():.3f}")
        
        # Identify column types
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        print(f"\nNumerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        
        # Encode categorical columns with CUSTOM ORDINAL ENCODING for lighting and weather
        print("\n" + "-"*80)
        print("ENCODING CATEGORICAL COLUMNS (with custom ordinal encoding)")
        print("-"*80)
        
        # Custom ordinal encoding for lighting (by risk level)
        if 'lighting' in self.categorical_cols:
            lighting_map = {'daylight': 1, 'dim': 2, 'night': 3}
            X['lighting'] = X['lighting'].map(lighting_map)
            self.label_encoders['lighting'] = lighting_map  # Store mapping
            print(f"Encoded 'lighting' (ordinal): daylight=1, dim=2, night=3")
            self.categorical_cols.remove('lighting')
            self.numerical_cols.append('lighting')
        
        # Custom ordinal encoding for weather (by risk level)
        if 'weather' in self.categorical_cols:
            weather_map = {'clear': 1, 'foggy': 2, 'rainy': 3}
            X['weather'] = X['weather'].map(weather_map)
            self.label_encoders['weather'] = weather_map  # Store mapping
            print(f"Encoded 'weather' (ordinal): clear=1, foggy=2, rainy=3")
            self.categorical_cols.remove('weather')
            self.numerical_cols.append('weather')
        
        # Standard label encoding for remaining categorical columns
        for col in self.categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            print(f"Encoded '{col}': {len(self.label_encoders[col].classes_)} unique values")
        
        # CREATE COMPOSITE FEATURE: my_score
        print("\n" + "-"*80)
        print("CREATING COMPOSITE FEATURE: my_score")
        print("-"*80)
        print("Formula: curvature × speed_limit × lighting × num_reported_accidents × weather")
        
        X['my_score'] = (
            X['curvature'] * 
            X['speed_limit'] * 
            X['lighting'] * 
            X['num_reported_accidents'] * 
            X['weather']
        )
        
        print(f"my_score statistics: min={X['my_score'].min():.2f}, max={X['my_score'].max():.2f}, "
              f"mean={X['my_score'].mean():.2f}, std={X['my_score'].std():.2f}")
        
        # Add to numerical columns list
        self.numerical_cols.append('my_score')
        
        # Compute correlations BEFORE scaling (for feature selection)
        print("\n" + "-"*80)
        print("COMPUTING FEATURE CORRELATIONS")
        print("-"*80)
        
        correlations = []
        for col in X.columns:
            corr = X[col].corr(y_train)
            correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        self.feature_correlations = dict(correlations)
        
        print("\nAll feature correlations with target:")
        for feature, corr in correlations:
            print(f"{feature:25} : {corr:8.4f}")
        
        # Filter features by correlation threshold
        self.selected_features = [feat for feat, corr in correlations 
                                 if abs(corr) >= correlation_threshold]
        
        print(f"\n" + "-"*80)
        print(f"FEATURE SELECTION (|correlation| >= {correlation_threshold})")
        print("-"*80)
        print(f"Features kept: {len(self.selected_features)}/{len(X.columns)}")
        print(f"Selected features: {self.selected_features}")
        
        removed_features = [feat for feat, corr in correlations 
                           if abs(corr) < correlation_threshold]
        if removed_features:
            print(f"\nRemoved features (low correlation):")
            for feat in removed_features:
                print(f"  - {feat}: {self.feature_correlations[feat]:.4f}")
        
        # Filter to selected features
        X = X[self.selected_features]
        
        # Scale numerical columns (only those in selected features)
        numerical_cols_selected = [col for col in self.numerical_cols 
                                  if col in self.selected_features]
        
        if numerical_cols_selected:
            print("\n" + "-"*80)
            print("SCALING NUMERICAL COLUMNS")
            print("-"*80)
            print(f"Scaling columns: {numerical_cols_selected}")
            X[numerical_cols_selected] = self.scaler.fit_transform(X[numerical_cols_selected])
            print("Numerical columns scaled using StandardScaler")
        
        print(f"\nFinal preprocessed data shape: {X.shape}")
        print("="*80)
        
        return X, y_train
    
    def transform(self, test_df):
        """
        Transform test data using fitted preprocessor.
        
        Parameters:
        -----------
        test_df : DataFrame
            Test data (without target column)
            
        Returns:
        --------
        X_test : DataFrame
            Preprocessed features
        """
        print("\n" + "="*80)
        print("TRANSFORMING TEST DATA")
        print("="*80)
        
        # Create a copy
        df = test_df.copy()
        
        # Drop id if present
        if 'id' in df.columns:
            X = df.drop(['id'], axis=1)
        else:
            X = df.copy()
        
        print(f"Original test data shape: {X.shape}")
        
        # Encode categorical columns using fitted encoders
        print("\n" + "-"*80)
        print("ENCODING CATEGORICAL COLUMNS")
        print("-"*80)
        
        # Custom ordinal encoding for lighting and weather (if they exist)
        if 'lighting' in X.columns and 'lighting' in self.label_encoders:
            X['lighting'] = X['lighting'].map(self.label_encoders['lighting'])
            print(f"Encoded 'lighting' using custom ordinal mapping")
        
        if 'weather' in X.columns and 'weather' in self.label_encoders:
            X['weather'] = X['weather'].map(self.label_encoders['weather'])
            print(f"Encoded 'weather' using custom ordinal mapping")
        
        # Standard encoding for remaining categorical columns
        for col in self.categorical_cols:
            if col in X.columns:
                # Handle unseen categories by mapping them to a known category
                def safe_transform(val):
                    if val in self.label_encoders[col].classes_:
                        return self.label_encoders[col].transform([val])[0]
                    else:
                        # Map unseen values to the most frequent class (index 0)
                        return 0
                
                X[col] = X[col].astype(str).apply(safe_transform)
                print(f"Encoded '{col}' using fitted encoder")
        
        # CREATE my_score feature
        print("\n" + "-"*80)
        print("CREATING COMPOSITE FEATURE: my_score")
        print("-"*80)
        
        X['my_score'] = (
            X['curvature'] * 
            X['speed_limit'] * 
            X['lighting'] * 
            X['num_reported_accidents'] * 
            X['weather']
        )
        print(f"my_score created for test data")
        
        # Select only the features that were selected during training
        print("\n" + "-"*80)
        print("SELECTING FEATURES")
        print("-"*80)
        X = X[self.selected_features]
        print(f"Selected {len(self.selected_features)} features")
        
        # Scale numerical columns using fitted scaler
        numerical_cols_selected = [col for col in self.numerical_cols 
                                  if col in self.selected_features]
        
        if numerical_cols_selected:
            print("\n" + "-"*80)
            print("SCALING NUMERICAL COLUMNS")
            print("-"*80)
            print(f"Scaling columns: {numerical_cols_selected}")
            X[numerical_cols_selected] = self.scaler.transform(X[numerical_cols_selected])
            print("Numerical columns scaled using fitted StandardScaler")
        
        print(f"\nFinal preprocessed test data shape: {X.shape}")
        print("="*80)
        
        return X
    
    def save(self, filepath='preprocessor.pkl'):
        """Save the preprocessor to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"\nPreprocessor saved to '{filepath}'")
    
    @staticmethod
    def load(filepath='preprocessor.pkl'):
        """Load a preprocessor from a file."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from '{filepath}'")
        return preprocessor


def preprocess_data(train_path='train.csv', test_path='test.csv', 
                   correlation_threshold=0.1, save_preprocessor=True):
    """
    Main preprocessing function for train and test data.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
    correlation_threshold : float
        Minimum absolute correlation to keep a feature
    save_preprocessor : bool
        Whether to save the preprocessor to disk
        
    Returns:
    --------
    X_train : DataFrame
        Preprocessed training features
    y_train : Series
        Training target
    X_test : DataFrame
        Preprocessed test features
    preprocessor : DataPreprocessor
        Fitted preprocessor object
    """
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize and fit preprocessor
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df, 
                                                   target_col='accident_risk',
                                                   correlation_threshold=correlation_threshold)
    
    # Transform test data
    X_test = preprocessor.transform(test_df)
    
    # Save preprocessor if requested (DISABLED - not needed)
    # if save_preprocessor:
    #     preprocessor.save('preprocessor.pkl')
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    return X_train, y_train, X_test, preprocessor


def train_random_forest_model(X_train, y_train, X_test_real, test_ids,
                             n_iter=50, cv_folds=3, test_size=0.15, 
                             val_size=0.15, random_state=42):
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : DataFrame
        Preprocessed training features
    y_train : Series
        Training target
    X_test_real : DataFrame
        Real test data for final predictions
    test_ids : array
        Test IDs for submission file
    n_iter : int
        Number of parameter settings sampled in RandomizedSearchCV
    cv_folds : int
        Number of cross-validation folds
    test_size : float
        Proportion of data for final test set
    val_size : float
        Proportion of remaining data for validation set
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    best_model : RandomForestRegressor
        Trained model with best hyperparameters
    results : dict
        Dictionary containing performance metrics
    """
    
    print("\n" + "="*80)
    print("RANDOM FOREST MODEL TRAINING")
    print("="*80)
    
    # Split data: train -> (train_dev + val + test_holdout)
    print("\n" + "-"*80)
    print("SPLITTING DATA")
    print("-"*80)
    
    # First split: separate final test set
    X_temp, X_test_holdout, y_temp, y_test_holdout = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from training set
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already split test
    X_train_dev, X_val, y_train_dev, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Training set (for hyperparameter search): {X_train_dev.shape}")
    print(f"Validation set (for evaluation during search): {X_val.shape}")
    print(f"Test holdout set (for final evaluation): {X_test_holdout.shape}")
    print(f"Real test set (for submission): {X_test_real.shape}")
    
    # Define hyperparameter search space
    print("\n" + "-"*80)
    print("HYPERPARAMETER SEARCH SPACE")
    print("-"*80)
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'max_samples': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    for param, values in param_distributions.items():
        print(f"{param:20}: {values}")
    
    print(f"\nTotal possible combinations: {np.prod([len(v) for v in param_distributions.values()]):,}")
    print(f"Sampling {n_iter} random combinations with {cv_folds}-fold cross-validation")
    
    # Initialize base model
    rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    # Perform randomized search
    print("\n" + "-"*80)
    print("RUNNING RANDOMIZED SEARCH (this may take a while...)")
    print("-"*80)
    
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=True
    )
    
    random_search.fit(X_train_dev, y_train_dev)
    
    # Get best model
    best_model = random_search.best_estimator_
    
    print("\n" + "-"*80)
    print("BEST HYPERPARAMETERS")
    print("-"*80)
    for param, value in random_search.best_params_.items():
        print(f"{param:20}: {value}")
    
    print(f"\nBest CV Score (neg_MSE): {random_search.best_score_:.6f}")
    print(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.6f}")
    
    # Evaluate on validation set
    print("\n" + "-"*80)
    print("VALIDATION SET PERFORMANCE")
    print("-"*80)
    
    y_val_pred = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"MSE:  {val_mse:.6f}")
    print(f"RMSE: {val_rmse:.6f}")
    print(f"MAE:  {val_mae:.6f}")
    print(f"R²:   {val_r2:.6f}")
    
    # Evaluate on test holdout set
    print("\n" + "-"*80)
    print("TEST HOLDOUT SET PERFORMANCE")
    print("-"*80)
    
    y_test_pred = best_model.predict(X_test_holdout)
    test_mse = mean_squared_error(y_test_holdout, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_holdout, y_test_pred)
    test_r2 = r2_score(y_test_holdout, y_test_pred)
    
    print(f"MSE:  {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE:  {test_mae:.6f}")
    print(f"R²:   {test_r2:.6f}")
    
    # Feature importance
    print("\n" + "-"*80)
    print("FEATURE IMPORTANCE")
    print("-"*80)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']:25}: {row['importance']:.6f}")
    
    # Retrain on full training data (train + val) for final predictions
    print("\n" + "-"*80)
    print("RETRAINING ON FULL TRAINING DATA")
    print("-"*80)
    
    X_full_train = pd.concat([X_train_dev, X_val], axis=0)
    y_full_train = pd.concat([y_train_dev, y_val], axis=0)
    
    print(f"Full training set size: {X_full_train.shape}")
    
    final_model = RandomForestRegressor(**random_search.best_params_, 
                                       random_state=random_state, 
                                       n_jobs=-1)
    final_model.fit(X_full_train, y_full_train)
    
    # Final evaluation on test holdout
    y_test_final_pred = final_model.predict(X_test_holdout)
    test_final_rmse = np.sqrt(mean_squared_error(y_test_holdout, y_test_final_pred))
    test_final_mae = mean_absolute_error(y_test_holdout, y_test_final_pred)
    test_final_r2 = r2_score(y_test_holdout, y_test_final_pred)
    
    print(f"\nFinal model performance on test holdout:")
    print(f"RMSE: {test_final_rmse:.6f}")
    print(f"MAE:  {test_final_mae:.6f}")
    print(f"R²:   {test_final_r2:.6f}")
    
    # Make predictions on real test data
    print("\n" + "-"*80)
    print("PREDICTING ON REAL TEST DATA")
    print("-"*80)
    
    y_test_real_pred = final_model.predict(X_test_real)
    
    print(f"Predictions generated: {len(y_test_real_pred)}")
    print(f"Prediction statistics:")
    print(f"  Min:  {y_test_real_pred.min():.6f}")
    print(f"  Max:  {y_test_real_pred.max():.6f}")
    print(f"  Mean: {y_test_real_pred.mean():.6f}")
    print(f"  Std:  {y_test_real_pred.std():.6f}")
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_ids,
        'accident_risk': y_test_real_pred
    })
    
    submission_file = 'submission_rf_myscore.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file saved to '{submission_file}'")
    print(f"Submission file shape: {submission.shape}")
    print("\nFirst few predictions:")
    print(submission.head(10))
    
    # Save the model (DISABLED - not needed)
    # model_file = 'random_forest_model.pkl'
    # with open(model_file, 'wb') as f:
    #     pickle.dump(final_model, f)
    # print(f"\nFinal model saved to '{model_file}'")
    
    # Compile results
    results = {
        'best_params': random_search.best_params_,
        'cv_rmse': np.sqrt(-random_search.best_score_),
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'test_rmse': test_final_rmse,
        'test_mae': test_final_mae,
        'test_r2': test_final_r2,
        'feature_importance': feature_importance,
        'submission_file': submission_file
    }
    
    return final_model, results


if __name__ == "__main__":
    # Step 1: Preprocess data
    print("="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    X_train, y_train, X_test_real, preprocessor = preprocess_data(
        train_path='train.csv',
        test_path='test.csv',
        correlation_threshold=0.1,
        save_preprocessor=True
    )
    
    # Load test IDs for submission
    test_df_raw = pd.read_csv('test.csv')
    test_ids = test_df_raw['id'].values
    
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"✓ Training features shape: {X_train.shape}")
    print(f"✓ Training target shape: {y_train.shape}")
    print(f"✓ Test features shape: {X_test_real.shape}")
    print(f"✓ Selected features: {list(X_train.columns)}")
    print(f"✓ Preprocessor saved to 'preprocessor.pkl'")
    
    # Step 2: Train Random Forest model
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80)
    
    model, results = train_random_forest_model(
        X_train=X_train,
        y_train=y_train,
        X_test_real=X_test_real,
        test_ids=test_ids,
        n_iter=50,
        cv_folds=3,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"✓ Best CV RMSE: {results['cv_rmse']:.6f}")
    print(f"✓ Validation RMSE: {results['val_rmse']:.6f}")
    print(f"✓ Test Holdout RMSE: {results['test_rmse']:.6f}")
    print(f"✓ Test Holdout R²: {results['test_r2']:.6f}")
    print(f"✓ Predictions saved to '{results['submission_file']}'")
    print("\n" + "="*80)
    print("ALL DONE! Ready for submission!")
    print("="*80)
