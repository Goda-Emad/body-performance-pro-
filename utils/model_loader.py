"""
Model Loader Module
-------------------
Handles loading trained machine learning models for the Body Performance Analytics application.
"""

import pickle
import joblib
import os
from typing import Dict, Any, Optional, Union
import streamlit as st

# مسارات النماذج
MODELS_DIR = 'models'

# أسماء ملفات النماذج
MODEL_FILES = {
    # Classification Models
    'knn': 'knn_model.pkl',
    'dt': 'dt_model.pkl',
    'svm_linear': 'svm_linear.pkl',
    'svm_rbf': 'svm_rbf.pkl',
    'mlp': 'mlp_classifier.pkl',
    
    # Regression Models
    'linear_regression': 'linear_regression.pkl',
    'dt_regressor': 'dt_regressor.pkl',
    'svr': 'svr_model.pkl',
    'mlp_regressor': 'mlp_regressor.pkl',
    
    # Preprocessing
    'scaler': 'scaler.pkl',
}


def load_pickle_model(file_path: str) -> Any:
    """
    Load a model from a pickle file.
    
    Parameters:
    -----------
    file_path : str
        Path to the pickle file
    
    Returns:
    --------
    Any
        Loaded model object
    """
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"Error loading pickle model from {file_path}: {str(e)}")


def load_joblib_model(file_path: str) -> Any:
    """
    Load a model from a joblib file.
    
    Parameters:
    -----------
    file_path : str
        Path to the joblib file
    
    Returns:
    --------
    Any
        Loaded model object
    """
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading joblib model from {file_path}: {str(e)}")


def load_single_model(model_name: str) -> Any:
    """
    Load a single model by name.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (key from MODEL_FILES)
    
    Returns:
    --------
    Any
        Loaded model object
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_FILES.keys())}")
    
    file_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    return load_pickle_model(file_path)


def load_models() -> Dict[str, Any]:
    """
    Load all trained models and scaler.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all models and scaler
    """
    models = {}
    
    for model_key, file_name in MODEL_FILES.items():
        file_path = os.path.join(MODELS_DIR, file_name)
        
        if os.path.exists(file_path):
            try:
                models[model_key] = load_pickle_model(file_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {model_key}: {e}")
                models[model_key] = None
        else:
            print(f"⚠️ Warning: Model file not found: {file_path}")
            models[model_key] = None
    
    # Check if all models loaded successfully
    missing_models = [k for k, v in models.items() if v is None]
    if missing_models:
        raise Exception(f"Failed to load models: {missing_models}")
    
    return models


def load_models_cached() -> Dict[str, Any]:
    """
    Load models with Streamlit caching for better performance.
    Use this in the main app.
    """
    return load_models()


def get_available_models() -> Dict[str, list]:
    """
    Get lists of available models by type.
    
    Returns:
    --------
    Dict[str, list]
        Dictionary with 'classification' and 'regression' model lists
    """
    classification = ['knn', 'dt', 'svm_linear', 'svm_rbf', 'mlp']
    regression = ['linear_regression', 'dt_regressor', 'svr', 'mlp_regressor']
    
    # Display names
    display_names = {
        'knn': 'K-Nearest Neighbors (k=9)',
        'dt': 'Decision Tree',
        'svm_linear': 'SVM (Linear Kernel)',
        'svm_rbf': 'SVM (RBF Kernel)',
        'mlp': 'Neural Network (MLP)',
        'linear_regression': 'Linear Regression',
        'dt_regressor': 'Decision Tree Regressor',
        'svr': 'Support Vector Regression (SVR)',
        'mlp_regressor': 'MLP Regressor (Neural Network)',
    }
    
    return {
        'classification': [(key, display_names.get(key, key)) for key in classification],
        'regression': [(key, display_names.get(key, key)) for key in regression],
    }


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    
    Returns:
    --------
    Dict[str, Any]
        Model information
    """
    model_info = {
        'knn': {
            'name': 'K-Nearest Neighbors',
            'type': 'classification',
            'params': {'n_neighbors': 9},
            'accuracy': 0.6316,
            'description': 'Simple distance-based classifier. Best k=9 found through tuning.'
        },
        'dt': {
            'name': 'Decision Tree',
            'type': 'classification',
            'params': {'max_depth': 10},
            'accuracy': 0.6745,
            'description': 'Tree-based classifier that splits data based on feature importance.'
        },
        'svm_linear': {
            'name': 'SVM (Linear Kernel)',
            'type': 'classification',
            'params': {'kernel': 'linear', 'C': 1.0},
            'accuracy': 0.6114,
            'description': 'Support Vector Machine with linear kernel.'
        },
        'svm_rbf': {
            'name': 'SVM (RBF Kernel)',
            'type': 'classification',
            'params': {'kernel': 'rbf', 'C': 1.0},
            'accuracy': 0.6887,
            'description': 'Support Vector Machine with RBF kernel - captures non-linear patterns.'
        },
        'mlp': {
            'name': 'Neural Network (MLP)',
            'type': 'classification',
            'params': {'hidden_layers': (64, 32), 'max_iter': 500},
            'accuracy': 0.7215,
            'description': 'Best performing classifier with 72.15% accuracy.'
        },
        'linear_regression': {
            'name': 'Linear Regression',
            'type': 'regression',
            'params': {},
            'r2': 0.7658,
            'rmse': 19.28,
            'description': 'Baseline linear model for broad jump prediction.'
        },
        'dt_regressor': {
            'name': 'Decision Tree Regressor',
            'type': 'regression',
            'params': {'max_depth': 10},
            'r2': 0.7221,
            'rmse': 21.00,
            'description': 'Tree-based regression model.'
        },
        'svr': {
            'name': 'Support Vector Regression',
            'type': 'regression',
            'params': {'kernel': 'rbf', 'C': 1.0},
            'r2': 0.7749,
            'rmse': 18.90,
            'description': 'SVM for regression tasks.'
        },
        'mlp_regressor': {
            'name': 'MLP Regressor',
            'type': 'regression',
            'params': {'hidden_layers': (64, 32), 'max_iter': 500},
            'r2': 0.7791,
            'rmse': 18.73,
            'description': 'Best performing regressor with R² = 0.7791.'
        }
    }
    
    return model_info.get(model_name, {'name': model_name, 'type': 'unknown'})


def check_models_integrity() -> Dict[str, bool]:
    """
    Check if all model files exist and can be loaded.
    
    Returns:
    --------
    Dict[str, bool]
        Dictionary with model names and their availability status
    """
    results = {}
    
    for model_key, file_name in MODEL_FILES.items():
        file_path = os.path.join(MODELS_DIR, file_name)
        
        if os.path.exists(file_path):
            try:
                model = load_pickle_model(file_path)
                results[model_key] = True
            except Exception:
                results[model_key] = False
        else:
            results[model_key] = False
    
    return results


if __name__ == "__main__":
    # Test the module
    print("Testing model_loader module...")
    
    try:
        # Check model files
        status = check_models_integrity()
        print("\n📁 Model files status:")
        for name, exists in status.items():
            print(f"  {'✅' if exists else '❌'} {name}")
        
        # Load models
        models = load_models()
        print(f"\n✅ Loaded {len(models)} models successfully")
        
        # Get available models
        available = get_available_models()
        print(f"\n📊 Classification models: {len(available['classification'])}")
        print(f"📈 Regression models: {len(available['regression'])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
