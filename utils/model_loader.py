"""
Model Loader Module
-------------------
Handles loading trained machine learning models.
"""

import pickle
import os
from typing import Dict, Any

# تعيين أسماء الملفات
MODEL_FILES = {
    'knn': 'knn_model.pkl',
    'dt': 'dt_model.pkl',
    'svm_linear': 'svm_linear.pkl',
    'svm_rbf': 'svm_rbf.pkl',
    'mlp': 'mlp_classifier.pkl',
    'scaler': 'scaler.pkl',
    'linear_regression': 'linear_regression.pkl',
    'dt_regressor': 'dt_regressor.pkl',
    'svr': 'svr_model.pkl',
    'mlp_regressor': 'mlp_regressor.pkl',
}


def load_models() -> Dict[str, Any]:
    """
    Load all trained models and scaler.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all models and scaler
    """
    models = {}
    
    # المسار المطلق للمشروع (مهما كان مكان التشغيل)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    print(f"DEBUG: Loading models from: {MODELS_DIR}")
    
    for model_key, file_name in MODEL_FILES.items():
        file_path = os.path.join(MODELS_DIR, file_name)
        
        # تحميل النموذج مباشرة (بدون try/except لإظهار الخطأ الحقيقي)
        with open(file_path, 'rb') as f:
            models[model_key] = pickle.load(f)
        print(f"✅ Loaded {model_key}")
    
    return models


def load_single_model(model_name: str):
    """Load a single model by name."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    file_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_available_models() -> Dict[str, list]:
    """Get lists of available models by type."""
    classification = ['knn', 'dt', 'svm_linear', 'svm_rbf', 'mlp']
    regression = ['linear_regression', 'dt_regressor', 'svr', 'mlp_regressor']
    
    display_names = {
        'knn': 'KNN (k=9)',
        'dt': 'Decision Tree',
        'svm_linear': 'SVM-Linear',
        'svm_rbf': 'SVM-RBF',
        'mlp': 'Neural Network (MLP)',
        'linear_regression': 'Linear Regression',
        'dt_regressor': 'Decision Tree Regressor',
        'svr': 'SVR',
        'mlp_regressor': 'MLP Regressor',
    }
    
    return {
        'classification': [(key, display_names.get(key, key)) for key in classification],
        'regression': [(key, display_names.get(key, key)) for key in regression],
    }


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    info = {
        'knn': {'name': 'KNN', 'type': 'classification', 'accuracy': 0.6316},
        'dt': {'name': 'Decision Tree', 'type': 'classification', 'accuracy': 0.6745},
        'svm_linear': {'name': 'SVM-Linear', 'type': 'classification', 'accuracy': 0.6114},
        'svm_rbf': {'name': 'SVM-RBF', 'type': 'classification', 'accuracy': 0.6887},
        'mlp': {'name': 'MLP', 'type': 'classification', 'accuracy': 0.7215},
        'linear_regression': {'name': 'Linear Regression', 'type': 'regression', 'r2': 0.7658},
        'dt_regressor': {'name': 'Decision Tree Regressor', 'type': 'regression', 'r2': 0.7221},
        'svr': {'name': 'SVR', 'type': 'regression', 'r2': 0.7749},
        'mlp_regressor': {'name': 'MLP Regressor', 'type': 'regression', 'r2': 0.7791},
        'scaler': {'name': 'StandardScaler', 'type': 'preprocessing'},
    }
    return info.get(model_name, {'name': model_name, 'type': 'unknown'})


def check_models_integrity() -> Dict[str, bool]:
    """Check if all model files exist."""
    results = {}
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    for model_key, file_name in MODEL_FILES.items():
        file_path = os.path.join(MODELS_DIR, file_name)
        results[model_key] = os.path.exists(file_path)
    return results
