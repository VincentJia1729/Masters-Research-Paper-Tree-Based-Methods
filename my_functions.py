import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import shap


from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

##################################################################3
# Random Forest

def time_series_cv_tuning_rf(X, y, param_grid, n_splits=5, print_results=True):
    """
    Perform time series cross-validation for hyperparameter tuning.
    
    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        Feature matrix
    y : pandas.Series or array-like
        Target variable
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter values to try
    n_splits : int, default=5
        Number of splits for TimeSeriesSplit
    print_results : bool, default=True
        Whether to print detailed results for each fold 
    
    Returns:
    --------
    tuple : (best_params, best_score)
        Best parameters and corresponding MSE score
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
        
    best_params = None
    best_score = float("inf")  # Initialize with infinity for MSE minimization
    
    # Generate all parameter combinations
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        fold_scores = []
        
        # Always print parameter combination being tested
        print(f"\nTesting parameters: {param_dict}")
        
        # Perform cross-validation for current parameter combination
        for split_train_index, split_test_index in tscv.split(X):
            # Split data for current fold
            split_X_train, split_X_test = X.iloc[split_train_index], X.iloc[split_test_index]
            split_y_train, split_y_test = y.iloc[split_train_index], y.iloc[split_test_index]
            
            # Train model with current parameters
            model = RandomForestRegressor(criterion='squared_error', verbose=0, **param_dict)
            model.fit(split_X_train, split_y_train)
            y_pred = model.predict(split_X_test)
            
            # Calculate metrics
            mse = mean_squared_error(split_y_test, y_pred)
            fold_scores.append(mse)
            
            # Print individual fold results ONLY if print_results is True
            if print_results:
                mape_score = mean_absolute_percentage_error(split_y_test, y_pred)
                print(f"  MSE: {mse:.4f}, MAPE: {mape_score:.2f}%")
        
        # Calculate average MSE across all folds
        avg_mse = np.mean(fold_scores)
        
        # Always print average MSE
        print(f"Avg MSE: {avg_mse:.4f}")
        
        # Update best parameters if current combination is better
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = param_dict
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best MSE: {best_score:.4f}")
    
    return best_params, best_score

###########################################
# XGBoost

def time_series_cv_tuning_xgboost(X, y, param_grid, n_splits=5, print_results=True):
    """
    Perform time series cross-validation for hyperparameter tuning.
    
    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        Feature matrix
    y : pandas.Series or array-like
        Target variable
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter values to try
    n_splits : int, default=5
        Number of splits for TimeSeriesSplit
    print_results : bool, default=True
        Whether to print detailed results for each fold 
    
    Returns:
    --------
    tuple : (best_params, best_score)
        Best parameters and corresponding MSE score
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
        
    best_params = None
    best_score = float("inf")  # Initialize with infinity for MSE minimization
    
    # Generate all parameter combinations
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        fold_scores = []
        
        # Always print parameter combination being tested
        print(f"\nTesting parameters: {param_dict}")
        
        # Perform cross-validation for current parameter combination
        for split_train_index, split_test_index in tscv.split(X):
            # Split data for current fold
            split_X_train, split_X_test = X.iloc[split_train_index], X.iloc[split_test_index]
            split_y_train, split_y_test = y.iloc[split_train_index], y.iloc[split_test_index]
            
            # Train model with current parameters
            model = xgb.XGBRegressor(objective = 'reg:squarederror', verbosity=0, **param_dict)
            model.fit(split_X_train, split_y_train)
            y_pred = model.predict(split_X_test)
            
            # Calculate metrics
            mse = mean_squared_error(split_y_test, y_pred)
            fold_scores.append(mse)
            
            # Print individual fold results ONLY if print_results is True
            if print_results:
                mape_score = mean_absolute_percentage_error(split_y_test, y_pred)
                print(f"  MSE: {mse:.4f}, MAPE: {mape_score:.2f}%")
        
        # Calculate average MSE across all folds
        avg_mse = np.mean(fold_scores)
        
        # Always print average MSE
        print(f"Avg MSE: {avg_mse:.4f}")
        
        # Update best parameters if current combination is better
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = param_dict
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best MSE: {best_score:.4f}")
    
    return best_params, best_score


###############################################
# Light GBM

def time_series_cv_tuning_lightgbm(X, y, param_grid, n_splits=5, print_results=True):
    """
    Perform time series cross-validation for hyperparameter tuning.
    
    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        Feature matrix
    y : pandas.Series or array-like
        Target variable
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter values to try
    n_splits : int, default=5
        Number of splits for TimeSeriesSplit
    print_results : bool, default=True
        Whether to print detailed results for each fold 
    
    Returns:
    --------
    tuple : (best_params, best_score)
        Best parameters and corresponding MSE score
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
        
    best_params = None
    best_score = float("inf")  # Initialize with infinity for MSE minimization
    
    # Generate all parameter combinations
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        fold_scores = []

        # special conditioning check for lightgbm
        if 'num_leaves' in param_dict and 'max_depth' in param_dict:
            if param_dict['max_depth'] != -1:  # Only apply constraint if depth is limited
                max_num_leaves = 2 ** param_dict['max_depth']
                param_dict['num_leaves'] = max(param_dict['num_leaves'], max_num_leaves)
                # set num_leaves default to 31
        
        # Always print parameter combination being tested
        print(f"\nTesting parameters: {param_dict}")
        
        # Perform cross-validation for current parameter combination
        for split_train_index, split_test_index in tscv.split(X):
            # Split data for current fold
            split_X_train, split_X_test = X.iloc[split_train_index], X.iloc[split_test_index]
            split_y_train, split_y_test = y.iloc[split_train_index], y.iloc[split_test_index]

            
            # Train model with current parameters
            model = lgb.LGBMRegressor(verbosity=-1, n_jobs=1, **param_dict)
            model.fit(split_X_train, split_y_train)
            y_pred = model.predict(split_X_test)
            
            # Calculate metrics
            mse = mean_squared_error(split_y_test, y_pred)
            fold_scores.append(mse)
            
            # Print individual fold results ONLY if print_results is True
            if print_results:
                mape_score = mean_absolute_percentage_error(split_y_test, y_pred)
                print(f"  MSE: {mse:.4f}, MAPE: {mape_score:.2f}%")
        
        # Calculate average MSE across all folds
        avg_mse = np.mean(fold_scores)
        
        # Always print average MSE
        print(f"Avg MSE: {avg_mse:.4f}")
        
        # Update best parameters if current combination is better
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = param_dict
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best MSE: {best_score:.4f}")
    
    return best_params, best_score

##############################################################
# Catboost 

def time_series_cv_tuning_catboost(X, y, param_grid, n_splits=5, print_results=True):
    """
    Perform time series cross-validation for hyperparameter tuning.
    
    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        Feature matrix
    y : pandas.Series or array-like
        Target variable
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter values to try
    n_splits : int, default=5
        Number of splits for TimeSeriesSplit
    print_results : bool, default=True
        Whether to print detailed results for each fold 
    
    Returns:
    --------
    tuple : (best_params, best_score)
        Best parameters and corresponding MSE score
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
        
    best_params = None
    best_score = float("inf")  # Initialize with infinity for MSE minimization
    
    # Generate all parameter combinations
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        fold_scores = []
        
        # Always print parameter combination being tested
        print(f"\nTesting parameters: {param_dict}")
        
        # Perform cross-validation for current parameter combination
        for split_train_index, split_test_index in tscv.split(X):
            # Split data for current fold
            split_X_train, split_X_test = X.iloc[split_train_index], X.iloc[split_test_index]
            split_y_train, split_y_test = y.iloc[split_train_index], y.iloc[split_test_index]
            
            # Train model with current parameters
            model = CatBoostRegressor(verbose=0, **param_dict)
            model.fit(split_X_train, split_y_train)
            y_pred = model.predict(split_X_test)
            
            # Calculate metrics
            mse = mean_squared_error(split_y_test, y_pred)
            fold_scores.append(mse)
            
            # Print individual fold results ONLY if print_results is True
            if print_results:
                mape_score = mean_absolute_percentage_error(split_y_test, y_pred)
                print(f"  MSE: {mse:.4f}, MAPE: {mape_score:.2f}%")
        
        # Calculate average MSE across all folds
        avg_mse = np.mean(fold_scores)
        
        # Always print average MSE
        print(f"Avg MSE: {avg_mse:.4f}")
        
        # Update best parameters if current combination is better
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = param_dict
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best MSE: {best_score:.4f}")
    
    return best_params, best_score


################################################################3
# Generic model evaluation

def evaluate_model_performance(model, X, y, model_name="Model", target_name="Target"):
    """
    Evaluate model performance with scatter plot and metrics.
    
    Parameters:
    -----------
    model : trained model object
        The trained model with a predict() method
    X : pandas DataFrame or numpy array
        Feature data for prediction
    y : pandas Series or numpy array
        Actual target values
    model_name : str, default="Model"
        Name of the model for display purposes
    target_name : str, default="Target" 
        Name of the target variable for display purposes
    
    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    """
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.6, color='royalblue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 45-degree line
    plt.xlabel(f"Actual Values ({target_name})")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    
    # Print metrics
    print(f"MSE of {model_name}: {mse:.4f}")
    print(f"R squared of {model_name}: {r2:.4f}")
    print(f"RMSE of {model_name}: {rmse:.4f}")
    print(f"MAE of {model_name}: {mae:.4f}")
    print(f"MAPE of {model_name}: {mape:.2f}%")
    
    # Return metrics as dictionary
    metrics = {
        'mse': mse,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return metrics

####################################################################

def shap_analysis_complete(model, X_train, X_test, model_name="Model", n_cols=3):
    """
    Perform comprehensive SHAP analysis with multiple visualization types.
    
    Parameters:
    -----------
    model : trained model object
        The trained model to explain
    X_train : pandas DataFrame
        Training data (used as background for SHAP explainer)
    X_test : pandas DataFrame  
        Test data to explain
    model_name : str, default="Model"
        Name of the model for plot titles
    n_cols : int, default=3
        Number of columns for dependence plot grid
    
    Returns:
    --------
    dict : Dictionary containing explainer and shap_values for further use
    """

    # Create explainer and calculate SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test, check_additivity=False)
    
    
    # 1. Beeswarm plot
    print("1. Beeswarm plot")
    shap.plots.beeswarm(shap_values)
    
    # 2. Bar plot
    print("2. Feature importance bar plot")
    shap.plots.bar(shap_values)
    
    # 3. Dependence plots in grid
    print("3. Dependence plots grid")
    n_features = len(X_test.columns)
    n_rows = math.ceil(n_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, name in enumerate(X_test.columns):
        plt.sca(axes[i])
        shap.dependence_plot(name, shap_values.values, X_test, 
                            show=False, ax=axes[i])
        axes[i].set_title(name, fontsize=10)
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{model_name}: SHAP Dependence Plots', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # 5. Heatmap ordered by predictions
    print("4. Heatmap ordered by predictions")
    y_pred = model.predict(X_test)
    order = np.argsort(y_pred)
    shap.plots.heatmap(shap_values, instance_order=order)
    

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'predictions': y_pred,
        'prediction_order': order
    }



