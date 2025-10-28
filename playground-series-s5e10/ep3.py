import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import warnings
import time
import itertools
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    #     preprocessor.save('preprocessor_nn.pkl')
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    return X_train, y_train, X_test, preprocessor


class NeuralNetwork(nn.Module):
    """
    PyTorch Neural Network for regression.
    """
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.3, use_batch_norm=True):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs, device, early_stopping=None):
    """
    Train the PyTorch model.
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss, model):
                if epoch > 0:  # Only print if not first epoch
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Restore best model
    if early_stopping is not None and early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    return train_losses, val_losses


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model and return predictions.
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
    
    return np.array(predictions), np.array(actuals)


def train_neural_network(X_train, y_train, X_test_real, test_ids,
                         test_size=0.15, val_size=0.15, random_state=42):
    """
    Train a PyTorch Neural Network model with hyperparameter tuning.
    
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
    test_size : float
        Proportion of data for final test set
    val_size : float
        Proportion of remaining data for validation set
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    best_model : nn.Module
        Trained model with best hyperparameters
    results : dict
        Dictionary containing performance metrics
    """
    
    print("\n" + "="*80)
    print("PYTORCH NEURAL NETWORK MODEL TRAINING")
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
    val_size_adjusted = val_size / (1 - test_size)
    X_train_dev, X_val, y_train_dev, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    # Convert to numpy arrays
    X_train_dev = X_train_dev.values.astype(np.float32)
    X_val = X_val.values.astype(np.float32)
    X_test_holdout = X_test_holdout.values.astype(np.float32)
    X_test_real = X_test_real.values.astype(np.float32)
    y_train_dev = y_train_dev.values.astype(np.float32).reshape(-1, 1)
    y_val = y_val.values.astype(np.float32).reshape(-1, 1)
    y_test_holdout = y_test_holdout.values.astype(np.float32).reshape(-1, 1)
    
    print(f"Training set: {X_train_dev.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test holdout set: {X_test_holdout.shape}")
    print(f"Real test set: {X_test_real.shape}")
    
    input_dim = X_train_dev.shape[1]
    
    # Define hyperparameter search space
    print("\n" + "-"*80)
    print("NEURAL NETWORK HYPERPARAMETER SEARCH SPACE")
    print("-"*80)
    
    param_grid = {
        'hidden_layers': [
            [128, 64],
            [128, 64, 32],
            [256, 128, 64]
        ],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [0.001, 0.002],
        'batch_size': [1024],
        'weight_decay': [0.001]
    }
    
    print("Search space:")
    for param, values in param_grid.items():
        print(f"  {param:20}: {len(values)} options")
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    total_configs = len(all_combinations)
    print(f"\nTotal configurations: {total_configs}")
    
    # Estimate time
    print("\n" + "-"*80)
    print("ESTIMATING TRAINING TIME")
    print("-"*80)
    
    # Run a quick timing test with smallest config
    print("Running timing test with smallest configuration...")
    test_config = {
        'hidden_layers': [64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 1024,
        'weight_decay': 0.001
    }
    
    # Create small dataset for timing
    sample_size = min(10000, len(X_train_dev))
    X_sample = X_train_dev[:sample_size]
    y_sample = y_train_dev[:sample_size]
    
    test_dataset = TensorDataset(torch.FloatTensor(X_sample), torch.FloatTensor(y_sample))
    test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=True)
    
    test_model = NeuralNetwork(
        input_dim=input_dim,
        hidden_layers=test_config['hidden_layers'],
        dropout_rate=test_config['dropout_rate'],
        use_batch_norm=True
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(test_model.parameters(), lr=test_config['learning_rate'])
    
    # Time one epoch
    start_time = time.time()
    test_model.train()
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = test_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    epoch_time = time.time() - start_time
    
    # Estimate full training time
    avg_epochs = 35  # Average with early stopping
    scale_factor = len(X_train_dev) / sample_size
    time_per_config = epoch_time * avg_epochs * scale_factor
    total_time_seconds = time_per_config * total_configs
    total_time_minutes = total_time_seconds / 60
    
    print(f"Timing test completed:")
    print(f"  Sample size: {sample_size}")
    print(f"  Time per epoch: {epoch_time:.2f}s")
    print(f"  Estimated epochs per config: {avg_epochs}")
    print(f"  Estimated time per config: {time_per_config:.1f}s")
    print(f"\n📊 TOTAL ESTIMATED TIME: {total_time_minutes:.1f} minutes ({total_time_seconds/3600:.1f} hours)")
    print(f"   for {total_configs} configurations")
    
    # Decide whether to use random search or grid search
    if total_configs > 50:
        # Use random search for large spaces
        print(f"\n⚡ Using RANDOMIZED SEARCH (sampling {min(50, total_configs)} configurations)")
        np.random.shuffle(all_combinations)
        selected_combinations = all_combinations[:min(50, total_configs)]
        estimated_time = time_per_config * len(selected_combinations) / 60
        print(f"   Estimated time: {estimated_time:.1f} minutes")
    else:
        # Use full grid search for small spaces
        print(f"\n⚡ Using FULL GRID SEARCH (all {total_configs} configurations)")
        selected_combinations = all_combinations
        estimated_time = total_time_minutes
        print(f"   Estimated time: {estimated_time:.1f} minutes")
    
    # Convert combinations to config dicts
    configs = []
    for combo in selected_combinations:
        config = {
            'hidden_layers': combo[0],
            'dropout_rate': combo[1],
            'learning_rate': combo[2],
            'batch_size': combo[3],
            'weight_decay': combo[4],
            'use_batch_norm': True,
            'epochs': 50
        }
        configs.append(config)
    
    # Try each configuration
    best_config = None
    best_val_rmse = float('inf')
    best_model = None
    
    print("\n" + "="*80)
    print(f"STARTING HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Total configurations to test: {len(configs)}")
    print(f"Estimated total time: {estimated_time:.1f} minutes")
    print()
    
    start_search_time = time.time()
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'='*80}")
        print(f"Architecture: {config['hidden_layers']}")
        print(f"Dropout: {config['dropout_rate']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Weight Decay: {config['weight_decay']}")
        print(f"Batch Size: {config['batch_size']}")
        
        config_start_time = time.time()
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_dev), 
            torch.FloatTensor(y_train_dev)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False
        )
        
        # Create model
        model = NeuralNetwork(
            input_dim=input_dim,
            hidden_layers=config['hidden_layers'],
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm']
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7
        )
        early_stopping = EarlyStopping(patience=10, verbose=False)
        
        # Train model
        print(f"Training...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            scheduler, config['epochs'], device, early_stopping
        )
        
        # Evaluate on validation set
        val_preds, val_actuals = evaluate_model(model, val_loader, device)
        val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
        val_mae = mean_absolute_error(val_actuals, val_preds)
        val_r2 = r2_score(val_actuals, val_preds)
        
        config_time = time.time() - config_start_time
        
        print(f"Validation RMSE: {val_rmse:.6f}")
        print(f"Validation MAE:  {val_mae:.6f}")
        print(f"Validation R²:   {val_r2:.6f}")
        print(f"Epochs trained:  {len(train_losses)}")
        print(f"Time taken: {config_time:.1f}s")
        
        # Check if this is the best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_config = config
            best_model = model
            print(f"✓ New best model!")
        
        # Show progress
        elapsed_time = time.time() - start_search_time
        remaining_configs = len(configs) - (i + 1)
        if i > 0:
            avg_time_per_config = elapsed_time / (i + 1)
            estimated_remaining = avg_time_per_config * remaining_configs / 60
            print(f"Progress: {i+1}/{len(configs)} | Elapsed: {elapsed_time/60:.1f}m | "
                  f"Remaining: ~{estimated_remaining:.1f}m")
    
    # Print best configuration
    total_search_time = time.time() - start_search_time
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*80)
    print(f"Total time: {total_search_time/60:.1f} minutes")
    print(f"Configurations tested: {len(configs)}")
    print(f"Average time per config: {total_search_time/len(configs):.1f}s")
    
    print("\n" + "="*80)
    print("BEST MODEL CONFIGURATION")
    print("="*80)
    print(f"Architecture: {best_config['hidden_layers']}")
    print(f"Dropout: {best_config['dropout_rate']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Weight Decay: {best_config['weight_decay']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Best Validation RMSE: {best_val_rmse:.6f}")
    
    # Print model summary
    print("\n" + "-"*80)
    print("MODEL SUMMARY")
    print("-"*80)
    print(best_model)
    total_params = sum(p.numel() for p in best_model.parameters())
    trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Evaluate on validation set
    print("\n" + "-"*80)
    print("VALIDATION SET PERFORMANCE")
    print("-"*80)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    val_preds, val_actuals = evaluate_model(best_model, val_loader, device)
    
    val_mse = mean_squared_error(val_actuals, val_preds)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(val_actuals, val_preds)
    val_r2 = r2_score(val_actuals, val_preds)
    
    print(f"MSE:  {val_mse:.6f}")
    print(f"RMSE: {val_rmse:.6f}")
    print(f"MAE:  {val_mae:.6f}")
    print(f"R²:   {val_r2:.6f}")
    
    # Evaluate on test holdout set
    print("\n" + "-"*80)
    print("TEST HOLDOUT SET PERFORMANCE")
    print("-"*80)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test_holdout), torch.FloatTensor(y_test_holdout))
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    test_preds, test_actuals = evaluate_model(best_model, test_loader, device)
    
    test_mse = mean_squared_error(test_actuals, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_actuals, test_preds)
    test_r2 = r2_score(test_actuals, test_preds)
    
    print(f"MSE:  {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE:  {test_mae:.6f}")
    print(f"R²:   {test_r2:.6f}")
    
    # Make predictions on real test data
    print("\n" + "-"*80)
    print("PREDICTING ON REAL TEST DATA")
    print("-"*80)
    
    best_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_real).to(device)
        y_test_real_pred = best_model(X_test_tensor).cpu().numpy().flatten()
    
    # Clip predictions to valid range [0, 1] since it's accident_risk
    y_test_real_pred = np.clip(y_test_real_pred, 0, 1)
    
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
    
    submission_file = 'submission_nn_optimized.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file saved to '{submission_file}'")
    print(f"Submission file shape: {submission.shape}")
    print("\nFirst few predictions:")
    print(submission.head(10))
    
    # Save the model (DISABLED - not needed)
    # model_file = 'neural_network_model.pth'
    # torch.save({
    #     'model_state_dict': best_model.state_dict(),
    #     'config': best_config,
    #     'input_dim': input_dim
    # }, model_file)
    # print(f"\nFinal model saved to '{model_file}'")
    
    # Compile results
    results = {
        'best_config': best_config,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'submission_file': submission_file
    }
    
    return best_model, results


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
    print(f"✓ Preprocessor saved to 'preprocessor_nn.pkl'")
    
    # Step 2: Train PyTorch Neural Network model
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80)
    
    model, results = train_neural_network(
        X_train=X_train,
        y_train=y_train,
        X_test_real=X_test_real,
        test_ids=test_ids,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"✓ Best Model: {results['best_config']['name']}")
    print(f"✓ Validation RMSE: {results['val_rmse']:.6f}")
    print(f"✓ Test Holdout RMSE: {results['test_rmse']:.6f}")
    print(f"✓ Test Holdout R²: {results['test_r2']:.6f}")
    print(f"✓ Predictions saved to '{results['submission_file']}'")
    print("\n" + "="*80)
    print("ALL DONE! Ready for submission!")
    print("="*80)
